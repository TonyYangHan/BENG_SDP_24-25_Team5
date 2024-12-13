import numpy as np
import skimage.feature
import skimage.filters
import skimage.segmentation
import skimage.morphology
import skimage.measure
# from Analyze import AnalyzeAndExport
from ImageHandling import LoadPILImages, ConvertImagesToPILImageStacks, SavePILImageStack, DrawRegionsOnImages
from Model import PrepareImagesForModel, Detect
from pathlib import Path

def SaveImages(data, suffix, pilImages, outputPath):
    stacks = ConvertImagesToPILImageStacks(data, pilImages)

    for stack, pilImage in zip(stacks, pilImages):
        p = Path(pilImage.filename)
        SavePILImageStack(stack, outputPath / (p.stem + suffix + p.suffix))


def SeparateContours(images: np.ndarray, edges: np.ndarray, foregroundThreshold: float,
                     gaussianSigma: float):
    separatedImages = np.zeros(images.shape, np.uint16)
    for i in range(images.shape[0]):
        # Consider organoids to be present at pixels with greater than 50% detection belief.
        foregroundMask = skimage.morphology.binary_opening(images[i] >= foregroundThreshold)

        # Watershed algorithm is used to distinguish organoids in contact. The algorithm needs a
        # heightmap and a set of initializer points (basins) for each organoid. The negated
        # detection image is used as the heightmap for watershed (i.e. the organoid centers, which
        # are the strongest predictions, should be at the lowest points in the heightmap).
        smoothForeground = skimage.filters.gaussian(images[i], gaussianSigma)
        heightmap = -smoothForeground

        # Basins are found by removing the organoid borders.
        centers = np.bitwise_and(foregroundMask, np.bitwise_not(edges[i]))
        basins = skimage.measure.label(centers)
        labeled = skimage.segmentation.watershed(heightmap, basins, mask=foregroundMask)

        # Some small organoids will be lost during the watershed if their edges were relatively too
        # thick to find their centers. Watershed should only really split organoids that are
        # touching, so we want to make sure that organoids in the original mask are preserved.
        #
        # First, find all regions that were lost during watershed.
        unsplit = np.logical_and(foregroundMask, labeled == 0)
        # Label the lost regions
        unsplit_labeled = skimage.measure.label(unsplit)
        # Make the label numbers for lost regions different from the watershed labels.
        unsplit_labeled[unsplit_labeled > 0] += labeled.max() + 1
        # Merge lost regions with the watershed labels.
        separatedImages[i] = labeled + unsplit_labeled
    return separatedImages


def Cleanup(images: np.ndarray, minimumArea: int, removeBorders: bool, fillHoles: bool):
    cleanedImages = np.zeros_like(images)
    print("Cleaning up objects...", end="", flush=True)
    for i in range(images.shape[0]):
        rps = skimage.measure.regionprops(images[i])
        for rp in rps:
            if rp.area < minimumArea:
                continue
            coords = np.asarray(rp.coords)
            if removeBorders and (0 in coords or
                                  images.shape[1]-1 in coords[:, 0] or
                                  images.shape[2]-1 in coords[:, 1]):
                continue
            mir, mic, mar, mac = rp.bbox
            cleanedImages[i, mir:mar, mic:mac] = np.where(
                rp.image_filled if fillHoles else rp.image, rp.label,
                cleanedImages[i, mir:mar, mic:mac])
    return cleanedImages


def DetectEdges(images: np.ndarray, gaussianSigma: float,
                hysteresisMinimum: float, hysteresisMaximum: float,
                foregroundThreshold: float):
    print("Detecting edges...", end="", flush=True)
    edgeImages = np.zeros(images.shape, dtype=bool)
    for i in range(images.shape[0]):
        # Reordered Canny edge detector (Sobel -> Gaussian -> Hysteresis threshold)
        smoothEdges = skimage.filters.gaussian(skimage.filters.sobel(images[i]), gaussianSigma)
        edges = skimage.filters.apply_hysteresis_threshold(smoothEdges, hysteresisMinimum,
                                                           hysteresisMaximum)

        foregroundMask = skimage.morphology.binary_opening(images[i] >= foregroundThreshold)
        edgeImages[i, :, :] = np.bitwise_and(edges, foregroundMask)
    return edgeImages


def RunPipeline(modelPath, imagePaths, outputPath,
                threshold: float, batchSize: int, edgeSigma: float, edgeMin: float,
                edgeMax: float, minimumArea: int, track: bool):
    model = LoadFullModel(modelPath)
    pilImages = LoadPILImages(imagePaths)
    preparedImages = PrepareImagesForModel(pilImages, model)
    detectionImages = Detect(model, preparedImages, batchSize)

    outputImages = {'Prepared Input': preparedImages}

    def Output(name: str, data):
        SaveImages(data, "_" + name.lower(), pilImages, outputPath)

    edgeImages = DetectEdges(detectionImages, edgeSigma, edgeMin, edgeMax, threshold)
    labeledImages = SeparateContours(detectionImages, edgeImages, threshold, edgeSigma)

    cleanedImages = Cleanup(labeledImages, minimumArea, True, True)

    ## Tracking
    # i = 0
    # stacks = ConvertImagesToStacks(cleanedImages, pilImages) if batch else [cleanedImages]
    # for stack in stacks:
    #     stack = Track(stack, 1, Inverse(Overlap))
    #     cleanedImages[i:(i + stack.shape[0])] = stack
    #     i += stack.shape[0]

    overlayImages = DrawRegionsOnImages(cleanedImages, preparedImages, (255, 255, 255), 16,
                                        (0, 255, 0))
    Output('Overlay', overlayImages)

    # # Analyze Images
    # stacks = ConvertImagesToStacks(cleanedImages, pilImages) if batch else [cleanedImages]
    # for stack, original in zip(stacks, pilImages):
    #     path = Path(original.filename)
    #     AnalyzeAndExport(stack, outputPath / (path.stem + "_data.xlsx"))

    return outputImages

mp,ip,op = "OrganoID/OptimizedModel", "../train/images", "../output_image"
RunPipeline(mp,ip,op,)