#!/usr/bin/env python3
import argparse
import glob
import json
import multiprocessing
from multiprocessing import dummy
import os
import traceback
os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs

import pathlib
import re
import subprocess
import sys
import time

name_remap = {
        # Int.
        '600': '600.perlbench_s',
        'perlbench_s': '600.perlbench_s',

        '602': '602.gcc_s',
        'gcc_s': '602.gcc_s',

        '605': '605.mcf_s',
        'mcf_s': '605.mcf_s',

        '620': '620.omnetpp_s',
        'omnetpp_s': '620.omnetpp_s',

        '623': '623.xalancbmk_s',
        'xalancbmk_s': '623.xalancbmk_s',

        '625': '625.x264_s',
        'x264_s': '625.x264_s',

        '631': '631.deepsjeng_s',
        'deepsjeng_s': '631.deepsjeng_s',

        '641': '641.leela_s',
        'leela_s': '641.leela_s',

        '648': '648.exchange2_s',
        'exchange2_s': '648.exchange2_s',

        '657': '657.xz_s',
        'xz_s': '657.xz_s',

        # FP.
        '619': '619.lbm_s',
        'lbm_s': '619.lbm_s',

        '621': '621.wrf_s',
        '621.wrf_s': '621.wrf_s',

        '638': '638.imagick_s',
        'imagick_s': '638.imagick_s',

        '644': '644.nab_s',
        'nab_s': '644.nab_s',

        '511': '511.povray_r',
        'povray_r': '511.povray_r',
        }

benchmark_path = {
        '600.perlbench_s': 'intspeed/perlbench.100M_32K.pp',
        '602.gcc_s': 'intspeed/gcc.100M_32K.pp',
        '605.mcf_s': 'intspeed/MCF.100M_32K.pp',
        '620.omnetpp_s': 'intspeed/omnetpp.100M_32K.pp',
        '623.xalancbmk_s': 'intspeed/xalancbmk.100M_32K.pp',
        '625.x264_s': 'intspeed/x264.100M_32K.pp',
        '631.deepsjeng_s': 'intspeed/deepsjeng.100M_32K.pp',
        '641.leela_s': 'intspeed/leela.100M_32K.pp',
        '648.exchange2_s': 'intspeed/exchange2.100M_32K.pp',
        '657.xz_s': 'intspeed/XZ.100M_32K.pp',

        '619.lbm_s': 'fpspeed/lbm.100M_32K.pp',
        #'621.wrf_s': 'fpspeed/wrf.100M_32K.pp',
        '638.imagick_s': 'fpspeed/imagick.100M_32K.pp',
        '644.nab_s': 'fpspeed/nab.100M_32K.pp',

        #'511.povray_r': 'fpspeed/povray.100M_32K.pp',
        }

# Groups:
#   1: Program name
#   2: Run name
#   3: Run number
#   4: Thread number
#   5: Region number
#   6: Warmup instructions
#   7: Prolog instructions
#   8: ROI instructions
#   9: Epilog instructions
#  10: Region number again
#  11: Thread number again
#  12: Region weight
# Example: gcc.try_33001_t0r5_warmup101500_prolog0_region100000000_epilog0_005_0-00092.0
details_re = re.compile('([^_]+)\.([^+]+)_(\d+)_t(\d+)r(\d+)_warmup(\d+)_prolog(\d+)_region(\d+)_epilog(\d+)_(\d{3})_(\d+)-(\d+).0')

class ConfigurationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def launch_benchmark(cmd):
    # Extract the output directory.
    output_arg = cmd[6]
    output_dir = output_arg.split(':')[0]
    print("Launching Batch job for {}".format(output_dir))
    completed = subprocess.run(cmd,
                stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE,
                 timeout=43200
                 )
    with open(os.path.join(output_dir, 'stdout'), 'w') as f:
        f.write(completed.stdout.decode('utf-8'))
    with open(os.path.join(output_dir, 'stderr'), 'w') as f:
        f.write(completed.stderr.decode('utf-8'))

class SniperLauncher:
    def __init__(self, simultaneity):
        self.simultaneity = simultaneity
        self.pool = multiprocessing.Pool(self.simultaneity, maxtasksperchild=4)

    def prepare_cmd(self, pinpoint_path, output_path, config_path):
        # Split out the pinpoint from the path.
        pinpoint_dir = os.path.dirname(pinpoint_path)
        pinpoint_name = os.path.basename(pinpoint_path)

        # Split out the config file from the path.
        config_dir = os.path.dirname(config_path)
        config_file = os.path.basename(config_path)

        # Prepare the output directory.
        name_parts = details_re.match(pinpoint_name)
        assert name_parts
        program_name        = name_parts.group(1)
        run_name            = name_parts.group(2)
        run_number          = name_parts.group(3)
        thread_number_short = name_parts.group(4)
        region_number_short = name_parts.group(5)
        warmup_insts        = name_parts.group(6)
        prolog_insts        = name_parts.group(7)
        roi_insts           = name_parts.group(8)
        epilog_insts        = name_parts.group(9)
        region_number_long  = name_parts.group(10)
        thread_number_long  = name_parts.group(11)
        region_weight       = name_parts.group(12)

        assert int(thread_number_short) == int(thread_number_long)
        assert int(region_number_short) == int(region_number_long)

        complete_output = os.path.join(output_path, region_number_long)
        pathlib.Path(complete_output).mkdir()

        # Write the region weight for convenience.
        with open(os.path.join(complete_output, 'weight'), 'w') as f:
            f.write('0.{}\n'.format(region_weight))

        cmd = ['docker', 'run', '--rm',
                '-v', '{}:/root/pinpoints'.format(pinpoint_dir),    # PinPoints root directory.
                '-v', '{}:/root/output'.format(complete_output),    # Output directory.
                '-v', '{}:/root/configs'.format(config_dir),        # Configuration directory.
                'sniper',
                'run-sniper',
                '--power',
                '-d', '/root/output',
                '-c', '/root/configs/{}'.format(config_file),
                '--pinballs', '/root/pinpoints/{}'.format(pinpoint_name)]
        return cmd

    def prepare_all_commands(self, benchmark, pinpoints, output, config, simultaneity):
        pinpoints_path = os.path.abspath(pinpoints)
        output_path = os.path.abspath(output)
        config_path = os.path.abspath(config)

        # Abort if the output path exists.  We don't want to accidentally overwrite existing data.
        assert not os.path.exists(output_path)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=False)

        # Finish the PinPoint path.
        benchmark_dir = benchmark_path[name_remap[benchmark]] if benchmark in name_remap else benchmark_path[benchmark]
        benchmark_points = glob.glob(os.path.join(pinpoints_path, benchmark_dir, '*.address'))
        benchmark_points = [bp[:-8] for bp in benchmark_points]

        cmds = []
        for bp in benchmark_points:
            cmd = self.prepare_cmd(bp, output_path, config_path)
            cmds.append(cmd)
        return cmds

    def batch_benchmark(self, benchmark, pinpoints, output, config, callback=None):
        cmds = self.prepare_all_commands(benchmark, pinpoints, output, config, None)
        async_result = self.pool.map_async(launch_benchmark, cmds, callback=callback)
        return async_result

def error_check(benchmark_dir):
    output_path = os.path.abspath(benchmark_dir)
    assert os.path.exists(output_path)
    regions = glob.glob(os.path.join(output_path, '*/'))
    regions = [region.rstrip('/') for region in regions]

    err_files = [os.path.join(region, 'stderr') for region in regions]

    # Examine the output to determine if simulation was successful.
    for err_file in err_files:
        with open(err_file) as f:
            result = f.read().strip()
        if len(result) == 0:
            return
        if '*Error*' in result:
            # Try to pull out the relevant part of the error message.
            msg = result[result.index('*Error* ')+8:]
            print('Configuration error in {}:'.format(err_file))
            print(msg)

def read_stats(stats_file):
    stats = {}
    with open(stats_file, 'r') as f:
        lines = f.readlines()

    key_stack = []
    for line in lines[1:]:
        parts = line.split('|')
        key = parts[0].strip()
        value = parts[1].strip()
        leading_spaces = len(parts[0]) - len(parts[0].lstrip())

        if leading_spaces == 0:
            del key_stack[:]
        elif leading_spaces == 2:
            del key_stack[1:]
        elif leading_spaces == 4:
            del key_stack[2:]
        else:
            assert False

        # Get to the proper place in the stats dict.
        current = stats
        for k in key_stack:
            current = current[k]

        if len(value) == 0:
            key_stack.append(key)
            current[key] = {}
            current = current[key]
        else:
            if value[-1] == '%':
                current[key] = float(value[:-1]) / 100.0
            else:
                try:
                    current[key] = int(value)
                except:
                    current[key] = float(value)

    return stats
def monitor_error(file):
    # open file and see if Error usind wildcard search
    error = False
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        keyword = ["Error","ValueError",
                    "[SNIPER] ERROR:", "errors",
                     "Invalid prefix roi-end"]
        # if any of the keywords are in the line, set error to true
        for word in keyword:
            if word in line:
                error = True
    return error

def combine_stats(output):
    try:
        output_path = os.path.abspath(output)
        assert os.path.exists(output_path)

        # Gather up the weights of the component regions.
        regions = glob.glob(os.path.join(output_path, '*/'))
        regions = [region.rstrip('/') for region in regions]

        # Region weight will be used for book keeping
        region_weight = {}
        # stores the original weight
        region_weight_orig = {}

        for region in regions:
            # check for error
            error_file = os.path.join(region, 'stderr')
            error = monitor_error(error_file)
            weight_file = os.path.join(region, 'weight')
            with open(weight_file, 'r') as f:
                weight = float(f.readline())
            if not error:
                region_weight[region.split('/')[-1]] = weight
                region_weight_orig[region.split('/')[-1]] = weight
            else:
                print("Error in region {}".format(region))
                print("Approximating the performance of the region")
                region_weight[region.split('/')[-1]] = 0.0
                region_weight_orig[region.split('/')[-1]] = weight

        # print the key with largest value
        max_key = max(region_weight, key=region_weight.get)
        

        # Gather up the stats.
        region_stats = {}
        for region in regions:
            # check if the sim.out file exists in region
            # else skip the region and read from region which has maximum weight
            stats_file = os.path.join(region, 'sim.out')
            if (os.path.exists(stats_file) and not region_weight[region.split("/")[-1]] == 0):
                region_stats[region.split('/')[-1]] = read_stats(stats_file)
            else:
                dummy_region = os.path.join(output_path, max_key)
                region_stats[region.split('/')[-1]] = read_stats(os.path.join(dummy_region, 'sim.out'))
            
        # Weight the stats appropriately.
        stats = {}

        # Time.
        stats['Time'] = 0.0
        for region in region_weight:
            weight = region_weight_orig[region]
            stats['Time'] += weight * region_stats[region]['Time (ns)']

        # IPC.
        stats['IPC'] = 0.0
        for region in region_weight:
            weight = region_weight_orig[region]
            stats['IPC'] += weight / region_stats[region]['IPC']
        stats['IPC'] = 1.0 / stats['IPC']

        # Branch prediction.
        stats['Branch Prediction'] = {}
        stats['Branch Prediction']['misprediction rate'] = 0.0
        stats['Branch Prediction']['MPKI'] = 0.0
        for region in region_weight:
            weight = region_weight_orig[region]
            stats['Branch Prediction']['misprediction rate'] += weight * region_stats[region]['Branch predictor stats']['misprediction rate']
            stats['Branch Prediction']['MPKI'] += weight * region_stats[region]['Branch predictor stats']['mpki']

        # TLB.
        stats['TLB'] = {}
        for tlb in region_stats['001']['TLB Summary']:
            stats['TLB'][tlb] = {}
            stats['TLB'][tlb]['miss rate'] = 0.0
            stats['TLB'][tlb]['MPKI'] = 0.0
        for region in region_weight:
            weight = region_weight_orig[region]
            for tlb in stats['TLB']:
                stats['TLB'][tlb]['miss rate'] += weight * region_stats[region]['TLB Summary'][tlb]['miss rate']
                stats['TLB'][tlb]['MPKI'] += weight * region_stats[region]['TLB Summary'][tlb]['mpki']

        # Cache.
        stats['Cache'] = {}
        for cache in region_stats['001']['Cache Summary']:
            stats['Cache'][cache] = {}
            stats['Cache'][cache]['miss rate'] = 0.0
            stats['Cache'][cache]['MPKI'] = 0.0
        for region in region_weight:
            weight = region_weight_orig[region]
            for cache in stats['Cache']:
                stats['Cache'][cache]['miss rate'] += weight * region_stats[region]['Cache Summary'][cache]['miss rate']
                stats['Cache'][cache]['MPKI'] += weight * region_stats[region]['Cache Summary'][cache]['mpki']
        
        
        # McPAT stats.
        region_mcpat_stats = {}
        for region in regions:
            #power_file = os.path.join(region, 'power.py')
            #print(power_file)
            sys.path.insert(0, region)
            try:  
                retry_count = 0
                while (retry_count < 10):
                    if(os.path.exists(os.path.join(region, 'power.py'))):
                        from power import power
                        break
                    else:
                        time.sleep(1)
                        retry_count += 1
            except ImportError:
                print('Error importing power.py')
            if(retry_count == 5):
                # read the power file from region which has highest weight
                power_file = os.path.join(output_path, max_key, 'power.py')
                print("Using Power.py from other region")
                sys.path.insert(0, power_file)
                from power import power
                region_mcpat_stats[region.split('/')[-1]] = power
            else:
                region_mcpat_stats[region.split('/')[-1]] = power
            del sys.path[0]

        stats['Power'] = {'Processor': {}}
        for region in region_mcpat_stats:
            weight = region_weight_orig[region]
            for stat in region_mcpat_stats[region]['Processor']:
                if stat not in stats['Power']['Processor']:
                    stats['Power']['Processor'][stat] = 0.0
                stats['Power']['Processor'][stat] += weight * region_mcpat_stats[region]['Processor'][stat]
    except Exception:
        print(traceback.format_exc()) 
    
    
    try:
        with open(os.path.join(output, 'stats.json'), 'w') as f:
            json.dump(stats, f, sort_keys=True, indent=2)
    except Exception as e:
        print('Error writing stats.json:')
        print(e)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', choices=['600', 'perlbench_s', '600.perlbench_s',
                                              '602', 'gcc_s', '602.gcc_s',
                                              '605', 'mcf_s', '605.mcf_s',
                                              '620', 'omnetpp', '620.omnetpp_s',
                                              '623', 'xalancbmk_s', '623.xalancbmk_s',
                                              '625', 'x264_s', '625.x264_s',
                                              '631', 'deepsjeng_s', '631.deepsjeng_s',
                                              '641', 'leela_s', '641.leela_s',
                                              '648', 'exchange2_s', '648.exchange2_s',
                                              '657', 'xz_s', '657.xz_s',

                                              ##'603', 'bwaves_s', '603.bwaves_s',
                                              ##'607', 'cactuBSSN_s', '607.cactuBSSN_s',
                                              '619', 'lbm_s', '619.lbm_s',
                                              '621', 'wrf_s', '621.wrf_s',
                                              ##'627', 'cam4_s', '627.cam4_s',
                                              ##'628', 'pop2_s', '628.pop2_s',
                                              '638', 'imagick_s', '638.imagick_s',
                                              '644', 'nab_s', '644.nab_s',
                                              ##'649', 'fotonik3d_s', '649.fotonik3d_s',
                                              ##'654', 'roms_s', '654.roms_s',

                                              '511', 'povray_r', '511.povray_r',
                                              ])
    parser.add_argument('-s', help='Path to Sniper SimPoints', default='./CPU2017')
    parser.add_argument('-c', help='Sniper configuration path', default='./config/gainestown.cfg')
    parser.add_argument('-d', help='Output directory', required=True)
    parser.add_argument('-n', help='Number of simultaneous jobs to launch', type=int, default=1)
    args = parser.parse_args()

    launcher = SniperLauncher(args.n)
    async_result = launcher.batch_benchmark(args.benchmark, args.s, args.d, args.c)
    async_result.wait()
    combine_stats(args.d)

if __name__ == '__main__':
    main()
