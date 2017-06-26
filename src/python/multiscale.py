import numpy as np
from scipy import ndimage
from scipy import misc
import skimage.transform
import os

class InvalidFlankerPosition(Exception):
# '''If flanker position is not possible raise this so we can deal with it by ignoring in convert_to_records. We try to fill complete grid of flanker eccs x target eccs, so we want to fail gracefully on invalid combinations of the two eccs'''
    pass

def make_square(im, digit_height):
    """Resize image so that the maximum dimension is crop_size, and paste into square"""
    x = np.asarray(im)
    h, w = x.shape
    output = skimage.img_as_ubyte(np.zeros((digit_height, digit_height))) 
    h, w = x.shape
    if h > w:
        new_w = int(w/float(h)*digit_height)
        x = misc.imresize(x, (digit_height, new_w))
        offset = int((digit_height - new_w)/2.0)
        output[:, offset:offset + new_w] = x
    else:
        new_h = int(h/float(w) * digit_height)
        x = misc.imresize(x, (new_h, digit_height))
        offset = int((digit_height - new_h)/2.0)
        output[offset:offset + new_h, :] = x
    assert len(x.shape) == 2, "Image not black and white"
    assert output.shape == (digit_height, digit_height)
    return output            

def pad_image(x, digit_height, field_size, ecc):
    field = skimage.img_as_ubyte(np.zeros((field_size, field_size)))
    half = field_size/2.0
    xmin = half - digit_height/2.0
    xmax = half + digit_height/2.0
    ymin = half - digit_height/2.0
    ymax = half + digit_height/2.0

    xmin = xmin + ecc
    xmax = xmax + ecc

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)

    assert field[ymin:ymax, xmin:xmax].shape == x.shape, ("Shape of x is {}"
        " and shape of place is {}, at ecc {}, {} field, {} dh".format(
            x.shape, field[ymin:ymax, xmin:xmax].shape, ecc, field_size, 
            digit_height))
    field[ymin:ymax, xmin:xmax] = x
    return field

def paste_on_background(x, background, digit_height, field_size, ecc):
    field =  misc.imresize(background, (field_size, field_size))
    half = field_size/2.0
    xmin = half - digit_height/2.0
    xmax = half + digit_height/2.0
    ymin = half - digit_height/2.0
    ymax = half + digit_height/2.0

    xmin = xmin + ecc
    xmax = xmax + ecc

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)

    assert field[ymin:ymax, xmin:xmax].shape == x.shape, ("Shape of x is {}"
        " and shape of place is {}, at ecc {}, {} field, {} dh".format(
            x.shape, field[ymin:ymax, xmin:xmax].shape, ecc, field_size, 
            digit_height))
    field[ymin:ymax, xmin:xmax] = x

    return field

def add_flanker(field, flanker, digit_height, ecc):
    '''Pastes flanker into image with target at ecc'''
    h, w = field.shape
    assert h == w, "Expected square image, got {} x {}".format(h, w)
    field_size = w

    flanker = make_square(flanker, digit_height)

    half = field_size/2.0
    xmin = half - digit_height/2.0
    xmax = half + digit_height/2.0
    ymin = half - digit_height/2.0
    ymax = half + digit_height/2.0

    xmin = xmin + ecc
    xmax = xmax + ecc

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)
 
    if field[ymin:ymax, xmin:xmax].shape != flanker.shape:
        raise InvalidFlankerPosition("Trying to paste wrong size flanker {} into crop area {}."
            "  Trying to paste flanker outside of image.\n"
            "  Max image ecc is {}, flanker ecc is {}".format(
            field[ymin:ymax, xmin:xmax].shape, flanker.shape,
            half, ecc))
    assert(field[ymin:ymax, xmin:xmax] == 0, "Trying to paste flanker onto a non-black image part")
    field[ymin:ymax, xmin:xmax] = np.maximum(flanker, field[ymin:ymax, xmin:xmax])
    return field

def build_crops(x, crop_size, field_size, num_scales, fovea, chevron=float('inf')):
    multiscale = np.zeros((num_scales, crop_size, crop_size), dtype=np.uint8)

    h, w = x.shape
    assert h == w, "Expected square images, got {} x {} ".format(h, w)

    for i in range(num_scales):
        if fovea == 'exponential':
            if num_scales == 1:
                size = field_size
            else:
                step = (field_size / float(crop_size)) ** (1 / float(num_scales - 1))
                size = crop_size * (step ** i)
        elif fovea == 'linear':
            if num_scales == 1:
                size = field_size
            else:
                step = int(field_size/(num_scales * float(crop_size)))
                size = crop_size * (step*i + 1)

        if (i - chevron) >= 0:
            null_scale = i - chevron
            if fovea == 'exponential':
                null_size = crop_size * (step ** null_scale)
            elif fovea == 'linear':
                null_size = crop_size * (null_scale + 1)

            x_blacked = blackout(x, null_size)
            crop = center_crop(x_blacked, size)
        else:
            crop = center_crop(x, size)

        crop = misc.imresize(crop, (crop_size, crop_size))

        multiscale[i, :, :] = crop

    return multiscale

def center_crop(x, size):
    h, w = x.shape
    assert h == w, "Function meant for square images only"
    half = h/2.0

    xmin = half - size/2.0
    xmax = half + size/2.0
    ymin = half - size/2.0
    ymax = half + size/2.0

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)

    crop = x[ymin:ymax, xmin:xmax] 
    return crop


def blackout(x, size):
    h, w = x.shape
    assert h == w, "Function meant for square images only"
    half = h/2.0

    xmin = half - size/2.0
    xmax = half + size/2.0
    ymin = half - size/2.0
    ymax = half + size/2.0

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)

    x[ymin:ymax, xmin:xmax] = 0
    return x

def divide_by_area_factor(ms, field_size, crop_size, num_scales, fovea ='exponential'):
    ms = ms.astype(np.float32)
    if fovea == 'exponential':
        if num_scales == 1:
            step = 1
        else:
            step = (field_size / float(crop_size)) ** (1 / float(num_scales - 1))

        for i in range(ms.shape[0]):
            crop = ms[i,:,:]
            if np.max(crop) == 0:
                continue
            crop = crop/float(np.max(crop)) / float(step ** (2 * (ms.shape[0]- i - 1)))
            crop = crop.astype(np.float32)
            ms[i,:,:] = crop
    elif fovea == 'linear':
        for i in range(ms.shape[0]):
            crop = ms[i,:,:]
            if np.max(crop) == 0:
                continue
            crop = crop/float(np.max(crop)) / float((ms.shape[0]- i)**2)
            crop = crop.astype(np.float32)
            ms[i,:,:] = crop

    return ms


def build_multiscale(im, crop_size, field_size, digit_height, num_scales, ecc, 
    fovea='exponential', 
    save=False, 
    save_directory=None, 
    save_basename='',
    flanker_type=0,
    flanker=None,
    flanker_ecc=None,
    flanker_height=None,
    chevron=float('inf'),
    contrast_norm='None',
    background=None,
    category=None):
    ''' Flanker type 0 means no flankers, 1 means one flanker, 2 means symmetrical flankers'''
    im = np.squeeze(im)
    if flanker is not None:
        flanker = np.squeeze(flanker)
    if flanker_height is None:
        flanker_height = digit_height

    assert len(im.shape) == 2, "Image is not grayscale, got {}".format(im.shape)
    assert isinstance(im, np.ndarray), "Image is not numpy ndarray"
    assert fovea in ('exponential', 'linear')
    if category is not None:
        assert background is not None, "Got empty background, but category is {}".format(category)
    if background is not None:
        assert category is not None, "Got a background image, but a None category"


    im = skimage.img_as_ubyte(im)
    
    x = make_square(im, digit_height)

    
    if background is not None:
        background = skimage.img_as_ubyte(background)
        image = paste_on_background(x, background, digit_height, field_size, ecc=ecc)
    else:
        image = pad_image(x, digit_height, field_size, ecc=ecc)
    
  
    if flanker_type == 1:
        if not (np.abs(ecc - flanker_ecc) >= (digit_height + flanker_height)/float(2)):
            raise InvalidFlankerPosition('Flanker wants to intersect target,'
                ' target ecc: {}, flanker ecc {}, target digit size: {}, flanker digit size: {}'.format(
                    ecc, flanker_ecc, digit_height, flanker_height))

        image = add_flanker(image, flanker, flanker_height, flanker_ecc)
    elif flanker_type == 2:
        if not (np.abs(ecc - flanker_ecc) >= (digit_height + flanker_height)/float(2)):
            raise InvalidFlankerPosition('Flanker wants to intersect target,'
                ' target ecc: {}, flanker ecc {}, target digit size: {}, flanker digit size: {}'.format(
                    ecc, flanker_ecc, digit_height, flanker_height))

        image = add_flanker(image, flanker, flanker_height, flanker_ecc)
        opp_ecc = 2 * ecc - flanker_ecc # symmetrical eccentricity from other flanker
        image = add_flanker(image, flanker, flanker_height, opp_ecc)
    else:
        assert flanker_type == 0, "Flanker type can only be 0, 1, 2, got {}".format(flanker_type)
    
    orig_image = image.copy()
    multiscale = build_crops(image, 
        crop_size, 
        field_size, 
        num_scales, 
        fovea=fovea, 
        chevron=chevron)

    if contrast_norm == 'areafactor':
        multiscale = divide_by_area_factor(multiscale, field_size, crop_size, num_scales, fovea=fovea)
    else:
        assert contrast_norm == 'None', "Can only have areafactor contrast norm or None"

    if save:
        fname = save_basename + \
        '_fovea_{}'.format(fovea[:3]) + \
        '_crsz_{}'.format(crop_size) + \
        '_fsz_{}'.format(field_size) + \
        '_dh_{}'.format(digit_height) + \
        '_nsc_{}'.format(num_scales) + \
        '_ecc_{}'.format(ecc) + \
        '_ftype_{}'.format(flanker_type) + \
        '_flanker_ecc_{}'.format(flanker_ecc) + \
        '_flankerheight_{}'.format(flanker_height) + \
        '_chevron_{}'.format(chevron) + \
        '_cnorm_{}'.format(contrast_norm) + \
        '_category_{}'.format(category[2:] if category is not None else None)
        misc.imsave(os.path.join(save_directory, fname) + '.png', orig_image)
        if contrast_norm == 'areafactor':
            # Dynamic range is too high for png, have to store as ??-bit tiff image instead.
            for i in range(num_scales):
                im = multiscale[i, :,:].copy()
                im = misc.toimage(im, mode='F')
                im.save(os.path.join(save_directory, fname + '_crop_{}'.format(i) + '.tiff'))
        else:
            for i in range(num_scales):        
                misc.imsave(os.path.join(save_directory, fname + '_crop_{}'.format(i) + '.png'), multiscale[i, :,:])
    assert multiscale.shape == (num_scales, crop_size, crop_size)

    return multiscale

