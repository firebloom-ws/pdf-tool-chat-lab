const DEFAULT_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073];
const DEFAULT_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711];

function computeResize(width, height, maxSide) {
  const scale = maxSide / Math.max(width, height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale))
  };
}

function resizeRgbaBilinear(source, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const output = new Uint8ClampedArray(targetWidth * targetHeight * 4);
  const xScale = sourceWidth / targetWidth;
  const yScale = sourceHeight / targetHeight;

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, (y + 0.5) * yScale - 0.5);
    const y0 = Math.max(0, Math.floor(sourceY));
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const yWeight = sourceY - y0;

    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, (x + 0.5) * xScale - 0.5);
      const x0 = Math.max(0, Math.floor(sourceX));
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const xWeight = sourceX - x0;
      const targetOffset = (y * targetWidth + x) * 4;

      for (let channel = 0; channel < 4; channel += 1) {
        const topLeft = source[(y0 * sourceWidth + x0) * 4 + channel];
        const topRight = source[(y0 * sourceWidth + x1) * 4 + channel];
        const bottomLeft = source[(y1 * sourceWidth + x0) * 4 + channel];
        const bottomRight = source[(y1 * sourceWidth + x1) * 4 + channel];
        const top = topLeft + (topRight - topLeft) * xWeight;
        const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
        output[targetOffset + channel] = Math.round(top + (bottom - top) * yWeight);
      }
    }
  }

  return output;
}

export class PixtralVisionPreprocessor {
  constructor(config = {}, preprocessorConfig = {}) {
    const visionConfig = config.vision_config ?? config;
    this.imageSize = visionConfig.image_size ?? 336;
    this.patchSize = visionConfig.patch_size ?? 14;
    this.imageMean = preprocessorConfig.image_mean ?? DEFAULT_IMAGE_MEAN;
    this.imageStd = preprocessorConfig.image_std ?? DEFAULT_IMAGE_STD;
  }

  prepare(image) {
    const { width, height, data } = image;
    const resized = computeResize(width, height, this.imageSize);
    const resizedPixels = resizeRgbaBilinear(
      data,
      width,
      height,
      resized.width,
      resized.height
    );

    const paddedWidth = Math.ceil(resized.width / this.patchSize) * this.patchSize;
    const paddedHeight = Math.ceil(resized.height / this.patchSize) * this.patchSize;
    const pixelValues = new Float32Array(3 * paddedWidth * paddedHeight);

    for (let y = 0; y < paddedHeight; y += 1) {
      for (let x = 0; x < paddedWidth; x += 1) {
        const paddedIndex = y * paddedWidth + x;
        const sourceX = Math.min(resized.width - 1, x);
        const sourceY = Math.min(resized.height - 1, y);
        const sourceOffset = (sourceY * resized.width + sourceX) * 4;

        for (let channel = 0; channel < 3; channel += 1) {
          const value = resizedPixels[sourceOffset + channel] / 255;
          pixelValues[channel * paddedWidth * paddedHeight + paddedIndex] =
            (value - this.imageMean[channel]) / this.imageStd[channel];
        }
      }
    }

    return {
      pixelValues,
      imageSize: [resized.height, resized.width],
      paddedSize: [paddedHeight, paddedWidth],
      patchGrid: {
        rows: paddedHeight / this.patchSize,
        cols: paddedWidth / this.patchSize
      },
      imageTokenCount: (paddedHeight / this.patchSize) * (paddedWidth / this.patchSize)
    };
  }
}
