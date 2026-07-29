/* Scales a grayscale PNG to an exact pixel height using Leptonica's own
 * pixScale, so that the h36 corpus arm goes through the same resampler
 * Tesseract will NOT have to run at recognition time (im_factor == 1.0).
 * Build: make -C testdata/golden/gen scaleline
 * Usage: scaleline <in.png> <height> <out.png>
 */
#include <stdio.h>
#include <stdlib.h>
#include <leptonica/allheaders.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <in.png> <height> <out.png>\n", argv[0]);
        return 2;
    }
    int target = atoi(argv[2]);
    if (target <= 0) {
        fprintf(stderr, "scaleline: bad height %s\n", argv[2]);
        return 2;
    }
    PIX *src = pixRead(argv[1]);
    if (src == NULL) {
        fprintf(stderr, "scaleline: cannot read %s\n", argv[1]);
        return 1;
    }
    PIX *gray = (pixGetDepth(src) == 8) ? pixClone(src) : pixConvertTo8(src, 0);
    float f = (float)target / (float)pixGetHeight(gray);
    PIX *dst = pixScale(gray, f, f);
    if (dst == NULL) {
        fprintf(stderr, "scaleline: pixScale failed\n");
        return 1;
    }
    /* Report what Leptonica actually produced; the caller checks it. */
    fprintf(stderr, "scaleline: %s %dx%d -> %dx%d (factor %g)\n", argv[1],
            pixGetWidth(gray), pixGetHeight(gray),
            pixGetWidth(dst), pixGetHeight(dst), f);
    if (pixWrite(argv[3], dst, IFF_PNG) != 0) {
        fprintf(stderr, "scaleline: cannot write %s\n", argv[3]);
        return 1;
    }
    pixDestroy(&dst);
    pixDestroy(&gray);
    pixDestroy(&src);
    return 0;
}
