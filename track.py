import argparse
import math
import matplotlib.animation as animation
import matplotlib.image as image
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
import numpy
import os
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import chan_vese
from skimage.feature import hog
import sys

class Parameters:
    video_name_list = ['kangaroo', 'gymnastics', 'butterfly', 'ball', 'ball_short']
    hog_cell_size = 8
    hog_direction = 9
    weak_classifier_num = 3
    K = 2
    debug = True

class Classifier:
    def __init__(self, sample, label):
        self.weak_classifiers = []
        self.error_maps = []
        weight = numpy.ones((sample.shape[0], 1)) / sample.shape[0]
        for i in range(0, Parameters.weak_classifier_num):
            solution = numpy.linalg.lstsq(sample, numpy.multiply(label, weight), rcond=None)
            classifier = {}
            classifier['classifier'] = solution[0]
            if Parameters.debug:
                print('[INFO] The weak classifier is: x = ' + numpy.array2string(classifier['classifier']))
            alpha, weight = self.compute_error(numpy.sign(numpy.matmul(sample, classifier['classifier'])), \
                                               label, weight)
            classifier['alpha'] = alpha
            self.weak_classifiers.append(classifier)
        
    def predict(self, sample):
        for i in range(0, Parameters.weak_classifier_num):
            self.weak_classifiers[i]['prediction'] = numpy.sign(numpy.matmul(sample, \
                                                                             self.weak_classifiers[i]['classifier']))
        final_result = numpy.sign(sum([self.weak_classifiers[i]['prediction'] * self.weak_classifiers[i]['alpha'] \
                                       for i in range(0, Parameters.weak_classifier_num)]))
        return final_result
        
    def update(self, sample, final_result):
        for i in range(0, Parameters.weak_classifier_num):
            self.weak_classifiers[i]['error'] = numpy.sum(numpy.absolute(self.weak_classifiers[i]['prediction'] - \
                                                                         final_result) / 2.0)
        self.weak_classifiers = sorted(self.weak_classifiers, key = lambda x: x['error']) 
        weight = numpy.ones((sample.shape[0], 1)) / sample.shape[0]
        for i in range(0, Parameters.K):
            alpha, weight = self.compute_error(self.weak_classifiers[i]['prediction'], final_result, weight)
            self.weak_classifiers[i]['alpha'] = alpha
        for i in range(Parameters.K, Parameters.weak_classifier_num):
            solution = numpy.linalg.lstsq(sample, numpy.multiply(final_result, weight), rcond=None)
            classifier = {}
            classifier['classifier'] = solution[0]
            alpha, weight = self.compute_error(numpy.sign(numpy.matmul(sample, classifier['classifier'])), \
                                               final_result, weight)
            classifier['alpha'] = alpha
            self.weak_classifiers[i] = classifier
        for classifier in self.weak_classifiers:
            classifier['prediction'] = []
            classifier['error'] = 0
        return final_result
    
    def compute_error(self, prediction, label, weight):
        error = numpy.absolute(prediction - label) / 2.0
        error_after_weight = numpy.sum(numpy.multiply(error, weight))
        alpha = 0.5 * math.log((1.0 - error_after_weight) / error_after_weight)
        weight = weight * numpy.exp(alpha * error)
        if Parameters.debug:
            print('[INFO] The status of the current weak classifier:\n' + \
                  '    ' + 'error: ' + str(error_after_weight) + '\n' + \
                  '    ' + 'weight: ' + str(alpha))
        return alpha, weight

class VideoTracker:
    def __init__(self, init_frame, init_mask):
        self.height = len(init_frame)
        self.width = len(init_frame[0])
        x1 = init_mask[0]
        y1 = init_mask[1]
        x2 = init_mask[2]
        y2 = init_mask[3]
        range_x1, range_y1, range_x2, range_y2 = self.enlarge_box(x1, y1, x2, y2)
        print('[INFO] Initialized sampling box at (' + str(range_x1) + ', ' + str(range_y1) + \
              ') -> (' + str(range_x2) + ', ' + str(range_y2) + ')')
        crop = numpy.array(init_frame[range_y1: range_y2, range_x1: range_x2, :])
        sample = self.extract_feature(crop)
        label_matrix = -numpy.ones((crop.shape[0], crop.shape[1]))
        label_matrix[y1 - range_y1: y2 - range_y1, x1 - range_x1: x2 - range_x1] = 1.0
        label = label_matrix.reshape(crop.shape[0] * crop.shape[1], 1)
        self.classifier = Classifier(sample, label)
        self.last_target_size = init_mask
        self.last_input_size = [range_x1, range_y1, range_x2, range_y2]
        '''
        # test prediction
        output = (self.classifier.predict(sample) + 1.0) / 2.0
        crop2 = output.reshape(crop.shape[0], crop.shape[1])
        # prediction confidence map
        # pyplot.figure()
        # pyplot.imshow(crop)
        pyplot.figure()
        pyplot.imshow(crop2, cmap='gray')
        pyplot.show()
        '''
        '''
        # Chan-Vese
        gray_image = numpy.dot(crop[..., :3], [0.2989, 0.5870, 0.1140])
        cv = chan_vese(gray_image, init_level_set=crop2, max_iter=10)
        pyplot.figure()
        pyplot.imshow(cv, cmap='gray')
        pyplot.show()
        '''
        '''
        # TODO
        # Active Contour
        center_x, center_y = self.get_center(crop2)
        radius = min(center_x, crop2.shape[1] - center_x, center_y, crop2.shape[0] - center_y)
        s = numpy.linspace(0, 2 * numpy.pi, 400)
        r = center_y + radius * numpy.sin(s)
        c = center_x + radius * numpy.cos(s)
        ac_init = numpy.array([r, c]).T
        ac = active_contour(crop2, ac_init,
                            # gaussian(crop, 3, preserve_range=False, multichannel=True), ac_init, \
                            coordinates='rc', \
                            alpha=0.15, \
                            beta=100, \
                            gamma=0.001, \
                            max_iterations=250, \
                            max_px_move=1.0, \
                            boundary_condition='periodic')
        figure, axis = pyplot.subplots()
        pyplot.imshow(crop)
        axis.plot(ac_init[:, 1], ac_init[:, 0], '--r', lw=3)
        axis.plot(ac[:, 1], ac[:, 0], '-b', lw=3)
        pyplot.show()
        '''
        
    def enlarge_box(self, x1, y1, x2, y2):
        mask_width = x2 - x1 + 1
        mask_height = y2 - y1 + 1
        range_x1 = max(0, x1 - mask_width)
        range_y1 = max(0, y1 - mask_height)
        range_x2 = min(self.width, x2 + mask_width)
        range_y2 = min(self.height, y2 + mask_height)
        new_wdith = range_x2 - range_x1
        new_height = range_y2 - range_y1
        tail_x = new_wdith % Parameters.hog_cell_size
        tail_y = new_height % Parameters.hog_cell_size
        return range_x1, range_y1, range_x2 - tail_x, range_y2 - tail_y
        
    def extract_feature(self, crop):
        # HOG features
        hog_result = hog(crop, \
                         orientations=Parameters.hog_direction, \
                         pixels_per_cell=(Parameters.hog_cell_size, Parameters.hog_cell_size), \
                         cells_per_block=(1, 1), \
                         # visualize=True, \
                         feature_vector=False,
                         multichannel=True)
        temp_crop = numpy.zeros((crop.shape[0], crop.shape[1], crop.shape[2] + Parameters.hog_direction))
        for i in range(0, len(temp_crop)):
            for j in range(0, len(temp_crop[0])):
                temp_crop[i][j][0: crop.shape[2]] = crop[i][j][:] / 255.0
                temp_crop[i][j][crop.shape[2]:] = hog_result[int(i / Parameters.hog_cell_size)][int(j / Parameters.hog_cell_size)][0][0][:]
        '''
        if Parameters.debug:
            print('[INFO] The HOG of each block is:')
            for i in range(0, len(hog_result)):
                for j in range(0, len(hog_result[0])):
                    print(hog_result[i][j][0][0][:])
        '''
        # reshape image to n * 3 and label to n * 1 matrices
        sample = temp_crop.reshape(temp_crop.shape[0] * temp_crop.shape[1], temp_crop.shape[2])
        # reversed the crop to image
        # crop2 = sample[:, 0: crop.shape[2]].reshape(crop.shape[0], crop.shape[1], crop.shape[2])
        # pyplot.figure()
        # pyplot.imshow(crop2)
        # pyplot.show()
        return sample
    
    def track(self, frame):
        range_x1 = self.last_input_size[0]
        range_y1 = self.last_input_size[1]
        range_x2 = self.last_input_size[2]
        range_y2 = self.last_input_size[3]
        crop = numpy.array(frame[range_y1: range_y2, range_x1: range_x2, :])
        sample = self.extract_feature(crop)
        output = self.classifier.predict(sample)
        output_mask = output.reshape(crop.shape[0], crop.shape[1])
        # Active Contour
        center_x, center_y = self.get_center(output_mask)
        radius = max(self.last_target_size[3] - self.last_target_size[1], \
                     self.last_target_size[2] - self.last_target_size[0]) / 2.0
        s = numpy.linspace(0, 2 * numpy.pi, 400)
        r = center_y + radius * numpy.sin(s)
        c = center_x + radius * numpy.cos(s)
        ac_init = numpy.array([r, c]).T
        ac = active_contour(gaussian(crop, 3, preserve_range=False, multichannel=True), ac_init, \
                            coordinates='rc', \
                            beta=10, \
                            alpha=0.015, \
                            gamma=0.001, \
                            max_iterations=100, \
                            max_px_move=0.1, \
                            boundary_condition='periodic')
        # update result size
        new_x1, new_y1, new_x2, new_y2 = self.get_box_from_ac(ac)
        self.last_target_size = [new_x1 + range_x1, new_y1 + range_y1, \
                                 new_x2 + range_x1, new_y2 + range_y1]
        new_range_x1, new_range_y1, new_range_x2, new_range_y2 = \
            self.enlarge_box(self.last_target_size[0], \
                             self.last_target_size[1], \
                             self.last_target_size[2], \
                             self.last_target_size[3])
        self.last_input_size = [new_range_x1, new_range_y1, \
                                new_range_x2, new_range_y2]
        label_matrix = -numpy.ones((crop.shape[0], crop.shape[1]))
        label_matrix[new_y1: new_y2, new_x1: new_x2] = 1.0
        label = label_matrix.reshape(crop.shape[0] * crop.shape[1], 1)
        self.classifier.update(sample, label)
        '''
        figure, axis = pyplot.subplots()
        pyplot.imshow(crop)
        axis.plot(ac_init[:, 1], ac_init[:, 0], '--r', lw=3)
        axis.plot(ac[:, 1], ac[:, 0], '-b', lw=3)
        pyplot.show()
        '''
        if Parameters.debug:
            print('[INFO] New track result at ' + str(self.last_target_size) + \
                  ' and new search range at ' + str(self.last_input_size))
        return ac[:, 0] + range_y1, ac[:, 1] + range_x1
    
    def get_center(self, matrix):
        average_i = 0
        average_j = 0
        count = 0
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                if matrix[i][j] > 0:
                    average_i += i
                    average_j += j
                    count += 1
        return average_j / count, average_i / count
        
    def get_box_from_ac(self, ac):
        x_coord = ac[:, 1]
        y_coord = ac[:, 0]
        x1 = max(0, int(numpy.min(x_coord)))
        y1 = max(0, int(numpy.min(y_coord)))
        x2 = min(self.width, int(numpy.max(x_coord)))
        y2 = min(self.height, int(numpy.max(y_coord)))
        return x1, y1, x2, y2

class VideoTrackerGui:
    def __init__(self, video_path):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.frame_paths = []
        for root, _, files in os.walk(video_path):
            for file in files:
                video_frame_path = os.path.join(root, file)
                self.frame_paths.append(video_frame_path)
        if not self.frame_paths:
            print('[ERROR] Video in ' + video_path + ' has no frames.')
            sys.exit(-1)
        self.figure, self.axis = pyplot.subplots()
        self.toggle_selector = widgets.RectangleSelector(self.axis, self.box_select_callback, \
                                                         drawtype='box', interactive=True)
        self.figure.canvas.mpl_connect('key_press_event', self.on_click)
        pyplot.axis("off")
        self.image_window = self.axis.imshow(image.imread(self.frame_paths[0]))
        pyplot.show()

    def box_select_callback(self, click, release):
        self.x1, self.y1 = int(click.xdata), int(click.ydata)
        self.x2, self.y2 = int(release.xdata), int(release.ydata)

    def animate(self, i):
        self.axis.clear()
        frame_path = self.frame_paths[i]
        frame = image.imread(frame_path)
        y, x = self.tracker.track(frame)
        self.image_window = self.axis.imshow(image.imread(self.frame_paths[0]))
        self.image_window.set_array(frame)
        self.axis.plot(x, y, '-b', lw=3)
        # pyplot.draw()
        return self.image_window

    def on_click(self, event):
        if event.key == 'enter':
            print('[INFO] Initialized bounding box at (' + str(self.x1) + ', ' + str(self.y1) + \
                  ') -> (' + str(self.x2) + ', ' + str(self.y2) + ')')
            self.toggle_selector.set_active(False)
            self.toggle_selector.set_visible(False)
            temp_frame = image.imread(self.frame_paths[0])
            self.tracker = VideoTracker(temp_frame, [self.x1, self.y1, self.x2, self.y2])
            # show the tracking result
            '''
            frame_plots = []
            for frame_path in self.frame_paths:
                frame = image.imread(frame_path)
                y, x = self.tracker.track(frame)
                frame_plot = pyplot.imshow(frame)
                frame_plots.append([frame_plot])
                p = pyplot.plot(x, y, '-b', lw=3)
                pyplot.draw()
                # pyplot.show()
            anime = animation.ArtistAnimation(self.figure, frame_plots, interval=20, blit=True, \
                                              repeat_delay=1000)
            '''
            anime = animation.FuncAnimation(self.figure, self.animate, frames=len(self.frame_paths), \
                                            interval=20, repeat=False)
            pyplot.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The implementation of the VOS algorithm in the \
                                                  attached paper.')
    parser.add_argument('video_name', help='The name of the video we want to track. Must be one of \
                                            the following name: ' + ' '.join(Parameters.video_name_list))
    args = parser.parse_args(sys.argv[1:])
    video_name = args.video_name
    if video_name not in Parameters.video_name_list:
        print('[ERROR] Input video name is not in the unit test list. Please refer to the help.')
        sys.exit(-1)
    current_path = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(current_path, 'unit_test', video_name)
    video_tracker_gui = VideoTrackerGui(video_path)
