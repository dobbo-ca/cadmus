// Regenerates Leptonica goldens. Manual step; see ../../../Makefile `goldens`.
// Each golden is a raw dump: width, height, depth as int32 LE, then packed rows.
#include <stdio.h>
#include <stdlib.h>
#include <leptonica/allheaders.h>

static void dump(const char *path, PIX *p) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int32_t hdr[3] = { pixGetWidth(p), pixGetHeight(p), pixGetDepth(p) };
    fwrite(hdr, sizeof(int32_t), 3, f);
    l_uint32 *data = pixGetData(p);
    int wpl = pixGetWpl(p);
    fwrite(data, sizeof(l_uint32), (size_t)wpl * pixGetHeight(p), f);
    fclose(f);
    printf("wrote %s (%dx%d d=%d)\n", path, hdr[0], hdr[1], hdr[2]);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: gen <input.png> <outdir>\n"); return 2; }
    PIX *src = pixRead(argv[1]);
    if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    PIX *gray = pixConvertTo8(src, 0);

    char path[512];
    snprintf(path, sizeof path, "%s/gray.bin", argv[2]);
    dump(path, gray);

    // Otsu, whole-image (no tiling), no smoothing.
    PIX *otsu = NULL;
    pixOtsuAdaptiveThreshold(gray, pixGetWidth(gray), pixGetHeight(gray), 0, 0, 0.0, NULL, &otsu);
    snprintf(path, sizeof path, "%s/otsu.bin", argv[2]);
    dump(path, otsu);

    // Sauvola over a 17x17 window (whsize 8), addborder=1 so Leptonica mirrors
    // the border itself. k is 11/32, which is exact in both float32 (the type
    // of Leptonica's factor) and float64 (the type of the Go k), so the two
    // implementations cannot disagree about the constant.
    PIX *sauvola = NULL;
    pixSauvolaBinarize(gray, 8, 0.34375f, 1, NULL, NULL, NULL, &sauvola);
    snprintf(path, sizeof path, "%s/sauvola.bin", argv[2]);
    dump(path, sauvola);

    pixDestroy(&src); pixDestroy(&gray); pixDestroy(&otsu); pixDestroy(&sauvola);
    return 0;
}
