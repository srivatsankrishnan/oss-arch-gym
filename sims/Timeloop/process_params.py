#!/usr/bin/env python3

import configparser
import json
import pandas as pd

class TimeloopConfigParams():
    def __init__(self, param_file) -> None:

        self.config_params = configparser.ConfigParser()
        try:
            self.config_params.read(param_file)
        except:
            print("Unable to read the parameters.")
            return

        # parameter dict
        self.arch_params = {}

        # PE Spatial array parameters
        self.arch_params['NUM_PEs']                                      = ['PE[0..{}]'.format(str(i-1)) for i in json.loads(self.config_params.get("PE Spatial Array", "nPEs"))]

        # MAC parameters
        self.arch_params['MAC_MESH_X']                                   = json.loads(self.config_params.get("MAC", "meshX"))

        # IFMap parameters
        self.arch_params['IFMAP_SPAD_CLASS']                             = json.loads(self.config_params.get("IFMap", "class"))
        self.arch_params['IFMAP_SPAD_ATRIBUTES']                         = {}
        self.arch_params['IFMAP_SPAD_ATRIBUTES']['memory_depth']         = json.loads(self.config_params.get("IFMap", "memory_depth"))
        self.arch_params['IFMAP_SPAD_ATRIBUTES']['block-size']           = json.loads(self.config_params.get("IFMap", "block-size"))
        self.arch_params['IFMAP_SPAD_ATRIBUTES']['read_bandwidth']       = json.loads(self.config_params.get("IFMap", "read_bandwidth"))
        self.arch_params['IFMAP_SPAD_ATRIBUTES']['write_bandwidth']      = json.loads(self.config_params.get("IFMap", "write_bandwidth"))
        #self.arch_params['IFMAP_SPAD_ATRIBUTES']['meshX']                = json.loads(self.config_params.get("MAC", "meshX"))

        # PSum parameters
        self.arch_params['PSUM_SPAD_CLASS']                              = json.loads(self.config_params.get("PSum", "class"))
        self.arch_params['PSUM_SPAD_ATRIBUTES']                          = {}
        self.arch_params['PSUM_SPAD_ATRIBUTES']['memory_depth']          = json.loads(self.config_params.get("PSum", "memory_depth"))
        self.arch_params['PSUM_SPAD_ATRIBUTES']['block-size']            = json.loads(self.config_params.get("PSum", "block-size"))
        self.arch_params['PSUM_SPAD_ATRIBUTES']['read_bandwidth']        = json.loads(self.config_params.get("PSum", "read_bandwidth"))
        self.arch_params['PSUM_SPAD_ATRIBUTES']['write_bandwidth']       = json.loads(self.config_params.get("PSum", "write_bandwidth"))
        #self.arch_params['PSUM_SPAD_ATRIBUTES']['meshX']                 = json.loads(self.config_params.get("MAC", "meshX"))

        # Weights parameters
        self.arch_params['WEIGHTS_SPAD_CLASS']                           = json.loads(self.config_params.get("Weights", "class"))
        self.arch_params['WEIGHTS_SPAD_ATRIBUTES']                       = {}
        self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['memory_depth']       = json.loads(self.config_params.get("Weights", "memory_depth"))
        self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['block-size']         = json.loads(self.config_params.get("Weights", "block-size"))
        self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['read_bandwidth']     = json.loads(self.config_params.get("Weights", "read_bandwidth"))
        self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['write_bandwidth']    = json.loads(self.config_params.get("Weights", "write_bandwidth"))
        #self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['meshX']              = json.loads(self.config_params.get("MAC", "meshX"))

        # Dummy buffer parameters
        self.arch_params['DUMMY_BUFFER_CLASS']                           = json.loads(self.config_params.get("DB", "class"))
        self.arch_params['DUMMY_BUFFER_ATTRIBUTES']                      = {}
        self.arch_params['DUMMY_BUFFER_ATTRIBUTES']['depth']             = json.loads(self.config_params.get("DB", "memory_depth"))
        self.arch_params['DUMMY_BUFFER_ATTRIBUTES']['block-size']        = json.loads(self.config_params.get("DB", "block-size"))

        # Shared Global Buffer parameters
        self.arch_params['SHARED_GLB_CLASS']                             = json.loads(self.config_params.get("SGB", "class"))
        self.arch_params['SHARED_GLB_ATTRIBUTES']                        = {}
        self.arch_params['SHARED_GLB_ATTRIBUTES']['memory_depth']        = json.loads(self.config_params.get("SGB", "memory_depth"))
        self.arch_params['SHARED_GLB_ATTRIBUTES']['n_banks']             = json.loads(self.config_params.get("SGB", "n_banks"))
        self.arch_params['SHARED_GLB_ATTRIBUTES']['block-size']          = json.loads(self.config_params.get("SGB", "block-size"))
        self.arch_params['SHARED_GLB_ATTRIBUTES']['read_bandwidth']      = json.loads(self.config_params.get("SGB", "read_bandwidth"))
        self.arch_params['SHARED_GLB_ATTRIBUTES']['write_bandwidth']     = json.loads(self.config_params.get("SGB", "write_bandwidth"))

        self._compute_param_sizes()
    
    def _compute_param_sizes(self):
        self.param_size = []
        self.param_size.append(len(self.arch_params['NUM_PEs']))
        self.param_size.append(len(self.arch_params['MAC_MESH_X']))
        self.param_size.append(len(self.arch_params['IFMAP_SPAD_CLASS']))
        self.param_size.append(len(self.arch_params['IFMAP_SPAD_ATRIBUTES']['memory_depth']))
        self.param_size.append(len(self.arch_params['IFMAP_SPAD_ATRIBUTES']['block-size']))
        self.param_size.append(len(self.arch_params['IFMAP_SPAD_ATRIBUTES']['read_bandwidth']))
        self.param_size.append(len(self.arch_params['IFMAP_SPAD_ATRIBUTES']['write_bandwidth']))
        #self.param_size.append(len(self.arch_params['IFMAP_SPAD_ATRIBUTES']['meshX']))
        self.param_size.append(len(self.arch_params['PSUM_SPAD_CLASS']))
        self.param_size.append(len(self.arch_params['PSUM_SPAD_ATRIBUTES']['memory_depth']))
        self.param_size.append(len(self.arch_params['PSUM_SPAD_ATRIBUTES']['block-size']))
        self.param_size.append(len(self.arch_params['PSUM_SPAD_ATRIBUTES']['read_bandwidth']))
        self.param_size.append(len(self.arch_params['PSUM_SPAD_ATRIBUTES']['write_bandwidth']))
        #self.param_size.append(len(self.arch_params['PSUM_SPAD_ATRIBUTES']['meshX']))
        self.param_size.append(len(self.arch_params['WEIGHTS_SPAD_CLASS']))
        self.param_size.append(len(self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['memory_depth']))
        self.param_size.append(len(self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['block-size']))
        self.param_size.append(len(self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['read_bandwidth']))
        self.param_size.append(len(self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['write_bandwidth']))
        #self.param_size.append(len(self.arch_params['WEIGHTS_SPAD_ATRIBUTES']['meshX']))
        self.param_size.append(len(self.arch_params['DUMMY_BUFFER_CLASS']))
        self.param_size.append(len(self.arch_params['DUMMY_BUFFER_ATTRIBUTES']['depth']))
        self.param_size.append(len(self.arch_params['DUMMY_BUFFER_ATTRIBUTES']['block-size']))
        self.param_size.append(len(self.arch_params['SHARED_GLB_CLASS']))
        self.param_size.append(len(self.arch_params['SHARED_GLB_ATTRIBUTES']['memory_depth']))
        self.param_size.append(len(self.arch_params['SHARED_GLB_ATTRIBUTES']['n_banks']))
        self.param_size.append(len(self.arch_params['SHARED_GLB_ATTRIBUTES']['block-size']))
        self.param_size.append(len(self.arch_params['SHARED_GLB_ATTRIBUTES']['read_bandwidth']))
        self.param_size.append(len(self.arch_params['SHARED_GLB_ATTRIBUTES']['write_bandwidth']))
    
    def get_param_size(self):
        return self.param_size
        
    def get_all_params(self):
        return self.arch_params

    def get_all_params_flattened(self, sep='.'):
        flat_params = {}
        for key, value in self.arch_params.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    flat_params[key + sep + key2] = value2
            else:
                flat_params[key] = value
        return flat_params

    def get_arch_param_template(self):
        
        arch_params = {
                        'SHARED_GLB_CLASS'       : 'smartbuffer_SRAM',
                        'SHARED_GLB_ATTRIBUTES'  : {'memory_depth': 16384, 'memory_width': 64, 'n_banks': 32, 'block-size': 4, 
                                                    'word-bits': 16, 'read_bandwidth': 16, 'write_bandwidth': 16},
                        'DUMMY_BUFFER_CLASS'     : 'SRAM',
                        'DUMMY_BUFFER_ATTRIBUTES': {'depth': 16, 'width': 16, 'word-bits': 16, 'block-size': 1},
                        'NUM_PEs'                : 'PE[0..167]',
                        'IFMAP_SPAD_CLASS'       : 'smartbuffer_SRAM',
                        'IFMAP_SPAD_ATRIBUTES'   : {'memory_depth': 16, 'memory_width': 16, 'block-size': 1, 'word-bits': 16, 
                                                     'read_bandwidth': 2, 'write_bandwidth': 2},
                        'WEIGHTS_SPAD_CLASS'     : 'smartbuffer_SRAM',
                        'WEIGHTS_SPAD_ATRIBUTES' : {'memory_depth': 192, 'memory_width': 16, 'block-size': 1, 'word-bits': 16, 
                                                     'read_bandwidth': 2, 'write_bandwidth': 2},
                        'PSUM_SPAD_CLASS'        : 'smartbuffer_SRAM',
                        'PSUM_SPAD_ATRIBUTES'    : {'memory_depth': 16, 'memory_width': 16, 'update_fifo_depth': 2, 'block-size': 1, 
                                                    'word-bits': 16, 'read_bandwidth': 2, 'write_bandwidth': 2},
                        'MAC_MESH_X'             : 14
                      }
        return arch_params