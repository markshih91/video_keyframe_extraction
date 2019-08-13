import os
import time
import argparse
import cv2
import torch
import numpy as np


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, imgs):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert imgs.ndim == 3, 'Image must be grayscale.'
        assert imgs.dtype == np.float32, 'Image must be float32.'
        N, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        inp = imgs.copy()
        inp = np.expand_dims(inp, axis=1)
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(N, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semis, coarse_descs = outs[0], outs[1]
        pts_list = []
        desc_list = []
        heatmap_list = []
        for idx in range(semis.shape[0]):
            semi = semis[idx].unsqueeze(0)
            coarse_desc = coarse_descs[idx].unsqueeze(0)
            # Convert pytorch -> numpy.
            semi = semi.data.cpu().numpy().squeeze()
            # --- Process points.
            dense = np.exp(semi)  # Softmax.
            dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
            # Remove dustbin.
            nodust = dense[:-1, :, :]
            # Reshape to get full resolution heatmap.
            Hc = int(H / self.cell)
            Wc = int(W / self.cell)
            nodust = nodust.transpose(1, 2, 0)
            heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
            heatmap = np.transpose(heatmap, [0, 2, 1, 3])
            heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
            xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
            if len(xs) == 0:
                pts_list.append(np.zeros((3, 0)))
                desc_list.append(None)
                heatmap_list.append(None)
                continue
            pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]
            pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.
            # Remove points along border.
            bord = self.border_remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]
            # --- Process descriptor.
            D = coarse_desc.shape[1]
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                # Interpolate into descriptor map using 2D point locations.
                samp_pts = torch.from_numpy(pts[:2, :].copy())
                samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
                samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
                samp_pts = samp_pts.transpose(0, 1).contiguous()
                samp_pts = samp_pts.view(1, 1, -1, 2)
                samp_pts = samp_pts.float()
                if self.cuda:
                    samp_pts = samp_pts.cuda()
                desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                desc = desc.data.cpu().numpy().reshape(D, -1)
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

            pts_list.append(pts)
            desc_list.append(desc)
            heatmap_list.append(heatmap)

        return pts_list, desc_list, heatmap_list


def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')

    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = 2 - 2 * np.clip(dmat, -1, 1)
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


class VideoStreamer(object):
    """ Class to help process image streams. Three types of possible inputs:"
      1.) USB Webcam.
      2.) A directory of images (files in directory matching 'img_glob').
      3.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, camid, h_ratio, w_ratio):
        self.cap = []
        self.video_file = False
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.i = 0
        self.num_frames = 0
        # If the "basedir" string is the word camera, then use a webcam.
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv2.VideoCapture(camid)
        else:
            # Try to open as a video.
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
                print('==> Processing Video Input.')
                self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_file = True

    def next_frame(self, subsample_rate):
        """ Return the next frame, and increment internal counter.
        Returns
           image: Next H x W image.
           status: True or False depending whether image was loaded.
        """
        if self.i == self.num_frames:
            return (None, None, 'max_len')
        ret, input_image = self.cap.read()
        if ret is False:
            return (None, None, False)

        input_image = cv2.resize(input_image,
                                 (int(input_image.shape[1] * subsample_rate),
                                  int(input_image.shape[0] * subsample_rate)))

        image_shape = input_image.shape[:2]
        patch_shape = np.asarray([int(input_image.shape[0] * self.h_ratio),
                                  int(input_image.shape[1] * self.w_ratio)])

        # center patch
        center_image_lu_pt = (image_shape - patch_shape) // 2
        center_image = input_image[
                       center_image_lu_pt[0]:center_image_lu_pt[0] + patch_shape[0],
                       center_image_lu_pt[1]:center_image_lu_pt[1] + patch_shape[1]].copy()
        center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2GRAY)

        # left-up patch
        lu_image = input_image[5:patch_shape[0]+5, 5:patch_shape[1]+5].copy()
        lu_image = cv2.cvtColor(lu_image, cv2.COLOR_RGB2GRAY)

        # right-up patch
        ru_image_lu_pt = np.array([5, image_shape[1] - patch_shape[1]-5])
        ru_image = input_image[
                   ru_image_lu_pt[0]:ru_image_lu_pt[0] + patch_shape[0],
                   ru_image_lu_pt[1]:ru_image_lu_pt[1] + patch_shape[1]].copy()
        ru_image = cv2.cvtColor(ru_image, cv2.COLOR_RGB2GRAY)

        # left-down patch
        ld_image_lu_pt = np.array([image_shape[0] - patch_shape[0]-5, 5])
        ld_image = input_image[
                   ld_image_lu_pt[0]:ld_image_lu_pt[0] + patch_shape[0],
                   ld_image_lu_pt[1]:ld_image_lu_pt[1] + patch_shape[1]].copy()
        ld_image = cv2.cvtColor(ld_image, cv2.COLOR_RGB2GRAY)

        # right-down patch
        rd_image_lu_pt = np.array([image_shape[0] - patch_shape[0]-5, image_shape[1] - patch_shape[1]-5])
        rd_image = input_image[
                   rd_image_lu_pt[0]:rd_image_lu_pt[0] + patch_shape[0],
                   rd_image_lu_pt[1]:rd_image_lu_pt[1] + patch_shape[1]].copy()
        rd_image = cv2.cvtColor(rd_image, cv2.COLOR_RGB2GRAY)

        patches = np.array([center_image, lu_image, ru_image, ld_image, rd_image])
        patches = patches.astype('float32') / 255.0
        # Increment internal counter.
        self.i = self.i + 1
        return (input_image, patches, True)


if __name__ == '__main__':

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Video Key Frames.')
    parser.add_argument('--input', type=str, default='test.mp4',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str, default='superpoint.pth',
                        help='Path to pretrained weights file (default: superpoint.pth).')
    parser.add_argument('--h_ratio', type=int, default=0.1,
                        help='ratio of the original image height (max: 0.33).')
    parser.add_argument('--w_ratio', type=int, default=0.2,
                        help='ratio of the original image width (max: 0.33).')
    parser.add_argument('--extract_dist', type=int, default=80,
                        help='Max distance for default save settings (default: 100).')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
                        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--min_matches', type=int, default=10,
                        help='Descriptor matching threshold (default: 10).')
    parser.add_argument('--match_interval', type=int, default=0,
                        help='Interval numbers of frames for compute matches (default: 0).')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
                        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--cuda', default=True, action='store_true',
                        help='Use cuda GPU to speed up network processing speed (default: True)')
    parser.add_argument('--display', action='store_true', default=True,
                        help='Display images to screen. (default: True).')
    parser.add_argument('--display_ratio', type=int, default=0.3,
                        help='display ratio of the original image size (default: 1).')
    parser.add_argument('--write', action='store_true', default=True,
                        help='Save output frames to a directory (default: True)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                        help='Directory where to write output frames (default: tracker_outputs/).')
    parser.add_argument('--write_subsample_rate', type=float, default=0.5,
                        help='Subsample rate of output frames (default: 0.5).')
    parser.add_argument('--video_id', type=int, default=1,
                        help='Video id for naming the image file to save (default: 0).')
    parser.add_argument('--start_img_id', type=int, default=0,
                        help='Started image id for naming the image file to save (default: 0).')
    opt = parser.parse_args()
    print(opt)

    # This class helps load input images from different sources.
    vs = VideoStreamer(opt.input, opt.camid, opt.h_ratio, opt.w_ratio)

    print('==> Loading pre-trained network.')
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,
                            conf_thresh=opt.conf_thresh,
                            nn_thresh=opt.nn_thresh,
                            cuda=opt.cuda)
    print('==> Successfully loaded pre-trained network.')

    # Create a window to display the demo.
    if opt.display:
        win = 'Original Video'
        cv2.namedWindow(win)
        win1 = 'Frame for save'
        cv2.namedWindow(win1)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Create output directory if desired.
    if opt.write:
        print('==> Will write outputs to %s' % opt.write_dir)
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)

    # frames for saving
    frames = []
    pts_list = []
    desc_list = []
    match_interval = 0

    print('==> Running.')
    while True:

        start = time.time()

        # Get a new image.
        img, patches, status = vs.next_frame(opt.write_subsample_rate)
        if status is False:
            print("read failed...")
            continue
        if status == 'max_len':
            break

        # Display visualization image to screen.
        if opt.display:
            cv2.imshow(win, cv2.resize(img, (int(img.shape[1] * opt.display_ratio),
                                             int(img.shape[0] * opt.display_ratio))))
            key = cv2.waitKey(opt.waitkey) & 0xFF
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break

        if len(frames) == 1 and len(pts_list) == 1 and len(desc_list) == 1 and match_interval < opt.match_interval:
            match_interval += 1
            continue

        end1 = time.time()
        pts = []
        desc = []
        none_num = 0

        res = fe.run(patches)
        pts = res[0]
        desc = res[1]
        for idx in range(len(pts)):
            if pts[idx] is None or desc[idx] is None:
                none_num += 1
        if none_num > 0:
            pts = [pts for pts in res[0] if pts is not None]
            desc = [desc for desc in res[1] if desc is not None]

        if none_num == len(patches):
            print('PointTracker: Warning, no points were added to some patches.')
            continue
        pts = np.concatenate(pts, axis=1)
        desc = np.concatenate(desc, axis=1)
        end2 = time.time()

        frames.append(img)
        pts_list.append(pts)
        desc_list.append(desc)

        if len(frames) == 2 and len(pts_list) == 2 and len(desc_list) == 2:
            matches = nn_match_two_way(desc_list[0], desc_list[1], opt.nn_thresh)

            if len(matches[0]) < opt.min_matches:
                frames.pop(0)
                pts_list.pop(0)
                desc_list.pop(0)
                continue

            key_pts1 = pts_list[0][:2, matches[0].astype(int)].transpose((1, 0))
            key_pts2 = pts_list[1][:2, matches[1].astype(int)].transpose((1, 0))

            distance = np.mean(np.sqrt(np.sum(np.square(key_pts1 - key_pts2), axis=1)))

            if distance >= opt.extract_dist:
                if opt.display:
                    cv2.imshow(win1, cv2.resize(frames[-1], (int(frames[-1].shape[1] * opt.display_ratio),
                                                             int(frames[-1].shape[0] * opt.display_ratio))))
                    key = cv2.waitKey(opt.waitkey) & 0xFF
                    if key == ord('p'):
                        print('Quitting, \'p\' pressed.')
                        break
                if opt.write:
                    out_file = os.path.join(opt.write_dir, '{0:0>3d}{1:0>6d}.jpg'.format(opt.video_id, opt.start_img_id))
                    print('Writing image to %s' % out_file)
                    cv2.imwrite(out_file, frames[-1])
                    opt.start_img_id += 1
                frames.pop(0)
                pts_list.pop(0)
                desc_list.pop(0)
            else:
                frames.pop(1)
                pts_list.pop(1)
                desc_list.pop(1)
        match_interval = 0

        if opt.display:
            end3 = time.time()
            preprocess_t = (1. / float(end1 - start))
            net_t = (1. / float(end2 - start))
            total_t = (1. / float(end3 - start))
            print('Processed image %d (pre_process: %.2f FPS, net: %.2f FPS, total: %.2f FPS).' \
                  % (vs.i, preprocess_t, net_t, total_t))

    # Close any remaining windows.
    cv2.destroyAllWindows()

    print('==> Finshed Demo.')
