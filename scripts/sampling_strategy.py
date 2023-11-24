import argparse
import logging
import re
from pathlib import Path
import pandas as pd

class SamplingStrategy:
    def __init__(self, inputPath, outputPath, annotationFilePath):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.annotationFilePath = annotationFilePath

    
    def stratifiedSampling(self):
        annotation_file = pd.read_csv(self.annotationFilePath)

        print(annotation_file)



strat = SamplingStrategy("self", "self", "ata/high_quality_fall_annotations.csv")

strat.stratifiedSampling()









