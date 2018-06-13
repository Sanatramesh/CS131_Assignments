import numpy as np
from scipy.ndimage.filters import convolve

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for imgi in range(Hi):
        for imgj in range(Wi):
            for ki in range(Hk):
                for kj in range(Wk):
                    x = imgi - ki + int(Hk/2)
                    y = imgj - kj + int(Wk/2)
                    if  x >= 0 and  y >= 0 and x < Hi and y < Wi:
                        out[imgi, imgj] += kernel[ki, kj] * image[x, y]

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:-pad_height, pad_width:-pad_width] = np.array(image, copy = True)
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image_pad = zero_pad(image, int(Hk/2), int(Wk/2))
    flipped_kernel = np.flip(np.flip(np.array(kernel, copy = True), axis = 1), axis = 0)

    for imgi in range(Hi):
        for imgj in range(Wi):
            out[imgi, imgj] = np.sum(image_pad[imgi:imgi+Hk, imgj:imgj+Wk] * flipped_kernel)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    out = convolve(image, kernel, mode='constant', cval=0.0)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE

    flipped_g = np.flip(np.flip(np.array(g, copy = True), axis = 1), axis = 0)
    out = conv_fast(f, flipped_g)

    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    flipped_g = np.flip(np.flip(np.array(g, copy = True), axis = 1), axis = 0) - np.mean(g)
    out = cross_correlation(f, flipped_g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    f_pad  = zero_pad(f, int(Hk/2), int(Wk/2))
    g_norm = (np.array(g, copy = True)- np.mean(g))/np.var(g)

    for imgi in range(Hi):
        for imgj in range(Wi):
            f_norm = f_pad[imgi:imgi+Hk, imgj:imgj+Wk]
            f_norm = (f_norm - np.mean(f_norm))/np.var(f_norm)
            out[imgi, imgj] = np.sum( f_norm * g_norm)
    ### END YOUR CODE

    return out
