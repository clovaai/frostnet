import numpy as np
import matplotlib.pyplot as plt
try:
    import tensorflow.compat.v1 as tf
    TENSORBOARD = True
except ImportError:
    print('no tensorflow found. set use_tensorboard = False')
    TENSORBOARD = False

try:
    import visdom
    VISDOM = True
except ImportError:
    print('no visdom found. set visdom_port = None')
    VISDOM = False


class Logger:
    def __init__(self, visdom_port=None, log_dir=None):
        if VISDOM and visdom_port:
            self.vis = visdom.Visdom(port=visdom_port)
            if not self.vis.check_connection():
                print('No visdom server found on port {}. set visdom_port = None'.format(visdom_port))
                self.vis = None
        else:
            self.vis = None
        self.use_visdom =  visdom_port
        self.use_tensorboard = True if TENSORBOARD and log_dir is not None else False

        if self.use_tensorboard:
            self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        if self.use_tensorboard:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)

    def image_summary(self, data, opts):
        if self.use_visdom:
            self.vis.images(data, opts=opts,)
            
    def histogram_summary(self, tag, values, step, bins=1000):
        if self.use_tensorboard:        
            """
            Logs the histogram of a list/vector of values.
            From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
            """

            # Create histogram using numpy
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill fields of histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))

            # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
            # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
            # Therefore we drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            # Create and write Summary
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            self.writer.add_summary(summary, step)  

def histogram_model(value,name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nbins = 200
    ranges = (-0.1,0.1)
    count = 0


    for layer in value:
        hist, bins = np.histogram(layer, bins=nbins, range=ranges)

        hist = hist/hist.sum()
        hist = np.clip(hist,0,0.05)
        xs = (bins[:-1] + bins[1:])/2
        color = plt.cm.twilight_shifted(count/len(value))
        if layer.min() == layer.max():
            hist = 0 
        ax.bar(xs, hist,zs=count,width=0.005, zdir = 'x', color= color, alpha=0.5)
        count+=1

    ax.set_xlabel('layer_index')
    ax.set_ylabel('magnitude')
    ax.set_zlabel('proportion')    
    ax.set_ylim(-0.1,0.1)
    plt.savefig(name)
