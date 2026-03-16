import numpy as np

def apply_fft_highpass(image_slice, radius_cutoff):
    """
    Applies a Gaussian High-Pass Filter (GHPF) in the frequency domain.
    
    This filter attenuates low-frequency components (smooth regions) while 
    preserving and enhancing high-frequency components (edges and fine details).
    Using a Gaussian curve prevents the "ringing" artifacts typical of ideal filters.

    Args:
        image_slice (np.ndarray): 2D input image array.
        radius_cutoff (float): Normalized cutoff frequency [0.0 - 1.0].

    Returns:
        np.ndarray: Filtered and normalized 2D image array.
    """
    if radius_cutoff <= 0.0:
        return image_slice
        
    # 1. Compute 2D Fast Fourier Transform and shift zero-frequency component to center
    f_transform = np.fft.fft2(image_slice)
    f_shift = np.fft.fftshift(f_transform)
    
    rows, cols = image_slice.shape
    center_row, center_col = rows // 2, cols // 2
    
    # 2. Create a spatial distance grid from the center
    Y, X = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((Y - center_row)**2 + (X - center_col)**2)
    
    # 3. Calculate absolute cutoff radius (D0)
    max_radius = min(center_row, center_col)
    D0 = radius_cutoff * max_radius
    
    # 4. Generate the Gaussian High-Pass Filter mask
    if D0 > 0:
        # Create a Gaussian low-pass base, then invert to create high-pass
        gaussian_low_pass = np.exp(-(distance_from_center**2) / (2 * (D0**2)))
        mask = 1.0 - gaussian_low_pass
    else:
        mask = np.ones((rows, cols))
    
    # 5. Apply the filter mask in the frequency domain
    f_shift_filtered = f_shift * mask
    
    # 6. Inverse FFT shift and compute the Inverse 2D FFT
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    
    # Extract magnitude (absolute value)
    img_back = np.abs(img_back)
    
    # 7. Normalize back to [0.0, 1.0] range
    if img_back.max() > 0:
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min())
        
    return img_back