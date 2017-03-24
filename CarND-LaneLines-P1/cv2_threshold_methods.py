    # initialize the list of threshold methods
    methods = [
        ("THRESH_BINARY", cv2.THRESH_BINARY),
        ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
        ("THRESH_TRUNC", cv2.THRESH_TRUNC),
        ("THRESH_TOZERO", cv2.THRESH_TOZERO),
        ("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]

    # loop over the threshold methods
    for (threshName, threshMethod) in methods:
        # threshold the image and show it
        (T, gray_adjust) = cv2.threshold(gray, 190, 255, threshMethod)
        plt.figure()
        plt.imshow(gray_adjust, cmap='gray')


