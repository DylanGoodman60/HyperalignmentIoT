import numpy as np
import cv2

## Align and procrustes functions modifyed from ContextLab
## https://github.com/ContextLab/hypertools/tree/master/hypertools/tools

def procrustes(source, target, scaling=True, reflection=True, reduction=False,
               oblique=False, oblique_rcond=-1):
    """
    Function to project from one space to another using Procrustean
    transformation (shift + scaling + rotation + reflection).
    The implementation of this function was based on the ProcrusteanMapper in
    pyMVPA: https://github.com/PyMVPA/PyMVPA
    See also: http://en.wikipedia.org/wiki/Procrustes_transformation
    Parameters
    ----------
    source : Numpy array
        Array to be aligned to target's coordinate system.
    target: Numpy array
        Source is aligned to this target space
    scaling : bool
        Estimate a global scaling factor for the transformation
        (no longer rigid body)
    reflection : bool
        Allow for the data to be reflected (so it might not be
        a rotation. Effective only for non-oblique transformations.
    reduction : bool
        If true, it is allowed to map into lower-dimensional
        space. Forward transformation might be suboptimal then and
        reverse transformation might not recover all original
        variance.
    oblique : bool
        Either to allow non-orthogonal transformation -- might
        heavily overfit the data if there is less samples than
        dimensions. Use `oblique_rcond`.
    oblique_rcond : float
        Cutoff for 'small' singular values to regularize the
        inverse. See :class:`~numpy.linalg.lstsq` for more
        information.
    Returns
    ----------
    aligned_source : Numpy array
        The array source is aligned to target and returned
    """
    def fit(source, target):
        datas = (source, target)
        sn, sm = source.shape
        tn, tm = target.shape

        # Check the sizes
        if sn != tn:
            raise ValueError("Data for both spaces should have the same " \
                  "number of samples. Got %d in template and %d in target space" \
                  % (sn, tn))

        # Sums of squares
        ssqs = [np.sum(d**2, axis=0) for d in datas]

        # XXX check for being invariant?
        #     needs to be tuned up properly and not raise but handle
        for i in range(2):
            if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                       * sn )**2)):
                raise ValueError("For now do not handle invariant in time datasets")

        norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]

        # add new blank dimensions to template space if needed
        if sm < tm:
            normed[0] = np.hstack( (normed[0], np.zeros((sn, tm-sm))) )

        if sm > tm:
            if reduction:
                normed[1] = np.hstack( (normed[1], np.zeros((sn, sm-tm))) )
            else:
                raise ValueError("reduction=False, so mapping from " \
                      "higher dimensionality " \
                      "template space is not supported. template space had %d " \
                      "while target %d dimensions (features)" % (sm, tm))

        source, target = normed
        if oblique:
            # Just do silly linear system of equations ;) or naive
            # inverse problem
            if sn == sm and tm == 1:
                T = np.linalg.solve(source, target)
            else:
                T = np.linalg.lstsq(source, target, rcond=oblique_rcond)[0]
            ss = 1.0
        else:
            # Orthogonal transformation
            # figure out optimal rotation
            U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                                     full_matrices=False)
            T = np.dot(Vh.T, U.T)

            if not reflection:
                # then we need to assure that it is only rotation
                # "recipe" from
                # http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
                # for more and info and original references, see
                # http://dx.doi.org/10.1007%2FBF02289451
                nsv = len(s)
                s[:-1] = 1
                s[-1] = np.linalg.det(T)
                T = np.dot(U[:, :nsv] * s, Vh)

            # figure out scale and final translation
            # XXX with reflection False -- not sure if here or there or anywhere...
            ss = sum(s)

        # if we were to collect standardized distance
        # std_d = 1 - sD**2

        # select out only relevant dimensions
        if sm != tm:
            T = T[:sm, :tm]

        # Assign projection
        if scaling:
            scale = ss * norms[1] / norms[0]
            proj = scale * T
        else:
            proj = T
        return proj

    def transform(data, proj):
        if proj is None:
            raise RuntimeError("Mapper needs to be trained before use.")

        d = np.asmatrix(data)

        # Do projection
        res = (d * proj).A

        return res

    # fit and transform
    proj = fit(source, target)
    return transform(source, proj)

def align(data):

##STEP 0: STANDARDIZE SIZE AND SHAPE##
    sizes_0 = [x.shape[0] for x in data]
    sizes_1 = [x.shape[1] for x in data]

    #find the smallest number of rows
    R = min(sizes_0)
    C = max(sizes_1)

    m = [np.empty((R,C), dtype=np.ndarray)] * len(data)

    for idx,x in enumerate(data):
        y = x[0:R,:]
        missing = C - y.shape[1]
        add = np.zeros((y.shape[0], missing))
        y = np.append(y, add, axis=1)
        m[idx]=y

    ##STEP 1: TEMPLATE##
    for x in range(0, len(m)):
        if x==0:
            template = np.copy(m[x])
        else:
            next = procrustes(m[x], template / (x + 1))
            template += next
    template /= len(m)

    ##STEP 2: NEW COMMON TEMPLATE##
    #align each subj to the template from STEP 1
    template2 = np.zeros(template.shape)
    for x in range(0, len(m)):
        next = procrustes(m[x], template)
        template2 += next
    template2 /= len(m)

    #STEP 3 (below): ALIGN TO NEW TEMPLATE
    aligned = [np.zeros(template2.shape)] * len(m)
    for x in range(0, len(m)):
        next = procrustes(m[x], template2)
        aligned[x] = next

    return aligned

## The same as alignment, but only for images
## Takes 2D numpy arrays of pixels representing images (one array per image)
## Returns alignment of the last image in the array
## If this is having errors, resort to align() function above
def align_images(images):
    ##STEP 1: TEMPLATE##
    for x in range(0, len(images)):
        if x==0:
            template = np.copy(images[x])
        else:
            next = procrustes(images[x], template / (x + 1))
            template += next
    template /= len(images)

    ##STEP 2: NEW COMMON TEMPLATE##
    #align each subj to the template from STEP 1
    template2 = np.zeros(template.shape)
    for x in range(0, len(images)):
        next = procrustes(images[x], template)
        template2 += next
    template2 /= len(images)

    ##STEP 3 (below): ALIGN TO NEW TEMPLATE
    j = len(images)-1
    next = procrustes(images[j], template2)

    return next


## Aligns a set of RGB images
## Takes a list of 3D numpy arrays representing R, G and B pixels of an image
## returns the alignment output of the images
def align_RGB_images(images):
    align_data_r = []
    align_data_g = []
    align_data_b = []

    for img in images:
        r, g, b = cv2.split(img)
        align_data_r.append(r)
        align_data_g.append(g)
        align_data_b.append(b)

    res_r = align_images(align_data_r)
    res_g = align_images(align_data_g)
    res_b = align_images(align_data_b)
    
    return cv2.merge((res_r, res_g, res_b))

def show_image(image):
    cv2.imshow('image', image/255)
    cv2.waitKey()

def save_image(filename, image):
    cv2.imwrite(filename, image)