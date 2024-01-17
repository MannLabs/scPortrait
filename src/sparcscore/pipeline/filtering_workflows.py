from sparcscore.pipeline.filter_segmentation import (
    SegmentationFilter,
    TiledSegmentationFilter
)

import numpy as np
from tqdm.auto import tqdm
import shutil
from collections import defaultdict

from sparcscore.processing.preprocessing import downsample_img_pxs

class BaseFiltering(SegmentationFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_unique_ids(self, mask):
        return(np.unique(mask)[1:])
    
    def return_empty_mask(self, input_image):
       #write out an empty entry
       self.save_classes(classes = {})

class filtering_match_nucleus_to_cytosol(BaseFiltering):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def match_nucleus_id_to_cytosol(self, nucleus_mask, cytosol_mask, return_ids_to_discard = False):
        all_nucleus_ids = self.get_unique_ids(nucleus_mask)
        all_cytosol_ids = self.get_unique_ids(cytosol_mask)
        
        nucleus_cytosol_pairs = {}
        nuclei_ids_to_discard = []

        for nucleus_id in tqdm(all_nucleus_ids):
            # get the nucleus and set the background to 0 and the nucleus to 1
            nucleus = (nucleus_mask == nucleus_id)
            
            # now get the coordinates of the nucleus
            nucleus_pixels = np.nonzero(nucleus)

            # check if those indices are not background in the cytosol mask
            potential_cytosol = cytosol_mask[nucleus_pixels]

            #if there is a cytosolID in the area of the nucleus proceed, else continue with a new nucleus
            if np.all(potential_cytosol != 0):

                unique_cytosol, counts = np.unique(
                    potential_cytosol, return_counts=True
                )
                all_counts = np.sum(counts)
                cytosol_proportions = counts / all_counts

                if np.any(cytosol_proportions >= self.config["filtering_threshold"]):
                    # get the cytosol_id with max proportion
                    cytosol_id = unique_cytosol[
                        np.argmax(cytosol_proportions >= self.config["filtering_threshold"])
                    ]
                    nucleus_cytosol_pairs[nucleus_id] = cytosol_id
                else:
                    #no cytosol found with sufficient quality to call so discard nucleus
                    nuclei_ids_to_discard.append(nucleus_id)

            else:
                #discard nucleus as no matching cytosol found
                nuclei_ids_to_discard.append(nucleus_id)
        
        #check to ensure that only one nucleus_id is assigned to each cytosol_id    
        cytosol_count = defaultdict(int)

        # Count the occurrences of each cytosol value
        for cytosol in nucleus_cytosol_pairs.values():
            cytosol_count[cytosol] += 1

        # Find cytosol values assigned to more than one nucleus and remove from dictionary
        multi_nucleated_nulceus_ids = []
        
        for nucleus, cytosol in nucleus_cytosol_pairs.items():
            if cytosol_count[cytosol] > 1:
                multi_nucleated_nulceus_ids.append(nucleus)
        
        #update list of all nuclei used
        nuclei_ids_to_discard.append(multi_nucleated_nulceus_ids) 
        
        #remove entries from dictionary
        # this needs to be put into a seperate loop because otherwise the dictionary size changes during loop and this throws an error
        for nucleus in multi_nucleated_nulceus_ids:
            del nucleus_cytosol_pairs[nucleus]  

        #get all cytosol_ids that need to be discarded
        used_cytosol_ids = set(nucleus_cytosol_pairs.values())
        not_used_cytosol_ids = set(all_cytosol_ids) - used_cytosol_ids
        not_used_cytosol_ids = list(not_used_cytosol_ids)
        
        if return_ids_to_discard:
            return(nucleus_cytosol_pairs, nuclei_ids_to_discard, not_used_cytosol_ids)
        else:
            return(nucleus_cytosol_pairs)

    def process(self, input_masks):
        
        if type(input_masks) == str:
            input_masks = self.read_input_masks(input_masks)

        #allow for optional downsampling to improve computation time
        if "downsampling_factor" in self.config.keys():
            N = self.config["downsampling_factor"]
            #use a less precise but faster downsampling method that preserves integer values
            input_masks = downsample_img_pxs(input_masks, N= N)

        #get input masks
        nucleus_mask = input_masks[0, :, :]
        cytosol_mask = input_masks[1, :, :]

        nucleus_cytosol_pairs = self.match_nucleus_id_to_cytosol(nucleus_mask, cytosol_mask)

        #save results
        self.save_classes(classes = nucleus_cytosol_pairs)

        #cleanup TEMP directories if not done during individual tile runs
        if hasattr(self, "TEMP_DIR_NAME"):
            shutil.rmtree(self.TEMP_DIR_NAME)

class multithreaded_filtering_match_nucleus_to_cytosol(TiledSegmentationFilter):
    method = filtering_match_nucleus_to_cytosol