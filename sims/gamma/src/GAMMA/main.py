from train import *
DEVELOP_MODE = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="latency", choices=('latency', 'energy', 'power', 'EDP', 'area'), help='First objective')
    parser.add_argument('--fitness2', type=str, default="energy", choices=('latency', 'energy', 'power', 'EDP', 'area'), help='Second objective')
    parser.add_argument('--num_pop', type=int, default=4,help='Number of populations')
    parser.add_argument('--parRS', default=False, action='store_true', help='Parallize across R S dimension')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs (i.e., Numbers of generations)')
    parser.add_argument('--outdir', type=str, default="outdir", help='Output directiory')
    parser.add_argument('--num_pe', type=int, default=1024, help='Number of PEs')
    parser.add_argument('--l1_size', type=int, default=-1, help='L1 size (local buffer size)')
    parser.add_argument('--l2_size', type=int, default=-1, help='L2 size (global buffer size)')
    parser.add_argument('--NocBW', type=int, default=-1, help='Network-on-Chip BW')
    parser.add_argument('--offchipBW', type=int, default=-1, help='Off-chip BW')
    parser.add_argument('--hwconfig', type=str, default=None, help='HW configuration file')
    parser.add_argument('--model', type=str, default="resnet18", help='Model to run')
    parser.add_argument('--num_layer', type=int, default=2, help='Number of layers to optimize')
    parser.add_argument('--singlelayer', type=int, default=0, help='The layer index to optimize')
    parser.add_argument('--slevel_min', type=int, default=2, help='Minimum number of parallelization level')
    parser.add_argument('--slevel_max', type=int, default=2, help='Maximum number of parallelization level')
    parser.add_argument('--fixedCluster', type=int, default=0, help='Rigid cluster size')
    parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    parser.add_argument('--costmodel_cstr', type=str, default='maestro_cstr', help='Constraint from Cost model')
    parser.add_argument('--mapping_cstr', type=str, default=None, help='Mapping constraint')
    parser.add_argument('--accel_cstr', type=str, default=None, help='Constraint from the HW type configuration of the accelerator under design')
    parser.add_argument('--area_budget', type=float, default=-1, help='The area budget (mm2). Set to -1 if no area upper-bound')
    parser.add_argument('--pe_limit', type=int, default=-1, help='Number of Processing Element budget. Set to -1 if no num_PE upper-bound')
    parser.add_argument('--use_factor', default=False, action='store_true', help='To only use factor as tile size.')
    parser.add_argument('--use_reorder',type = str, default='True', help='To use reorder operation')
    parser.add_argument('--reorder_alpha', type=float, default=0.9, help='The alpha for reorder operation')
    parser.add_argument('--use_growing', type=str, default='True', help='To use growing operation')
    parser.add_argument('--growing_alpha', type=float, default=0.4, help='The alpha for reorder operation')
    parser.add_argument('--use_aging', type=str,default='True', help='To use shrinking operation')
    parser.add_argument('--aging_alpha', type=float, default=0.5, help='The alpha for shrinking operation')
    
    opt = parser.parse_args()
    opt = set_hw_config(opt)
    if DEVELOP_MODE:
        history_path = "/usr/scratch/felix/my_code/history/gamma_flex/"
        if os.path.exists(history_path) is False:
            history_path = "/home/felix/Documents/2019summer/history/gamma_flex/"
            if os.path.exists(history_path) is False:
                history_path = "/Users/chuchu/Documents/gt_local/history/gamma_flex/"
    else:
        history_path = '../../'

    m_file_path = "../../data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    if opt.singlelayer:
        model_defs=model_defs[opt.singlelayer-1:opt.singlelayer]
    else:
        if opt.num_layer:
            model_defs = model_defs[:opt.num_layer]
    _, dim_size = model_defs.shape
    now = datetime.now()
    now_date = "{}".format(now.date())
    now_time = "{}".format(now.time())
    outdir = opt.outdir
    outdir = os.path.join(history_path, outdir)
    cstr_name = get_cstr_name(mapping_cstr=opt.mapping_cstr)
    exp_name = f"GAMMA_{opt.model}{f'-Lay{opt.singlelayer}' if opt.singlelayer>0 else ''}{f'-nLay{opt.num_layer}' if opt.singlelayer<1 and opt.num_layer>0 else ''}_SL-{opt.slevel_min}-{opt.slevel_max}" \
               f"{f'_FixCl-{opt.fixedCluster}' if opt.fixedCluster>0 else ''}_F1-{opt.fitness1}_GEN-{ opt.epochs}_POP-{opt.num_pop}_Area-{opt.area_budget}_MaxPEs-{opt.pe_limit}" \
               f"{f'_FixedPE-{opt.num_pe}' if opt.num_pe>0 else ''}{f'_L2Size-{opt.l2_size}' if opt.l2_size>0 else ''}" \
               f"{f'_L1Size-{opt.l1_size}' if opt.l1_size>0 else ''}{'_factorOnly' if opt.use_factor else ''}{f'_CostModelCstr-{opt.costmodel_cstr}' if opt.costmodel_cstr else ''}" \
               f"{f'_UseReorder' if opt.use_reorder=='True' else ''}{f'_UseGrowing' if opt.use_growing=='True' else ''}{f'_UseAging' if opt.use_aging=='True' else ''}" \
                f"{f'_ReorderAlpha-{opt.reorder_alpha}' if opt.use_reorder=='True' else ''}{f'_GrowingAlpha-{opt.growing_alpha}' if opt.use_growing=='True' else ''}" \
                f"{f'_AgingAlpha-{opt.aging_alpha}' if opt.use_aging=='True' else ''}" 
    outdir_exp = os.path.join(outdir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_exp, exist_ok=True)
    chkpt_file_t = "{}".format("result")
    chkpt_file = os.path.join(outdir_exp, chkpt_file_t + "_c.plt")
    map_cstr = None
    if opt.accel_cstr:
        accel_file = importlib.import_module(f'data.mapping_cstr.advanced_cstr.accel_cstr.{opt.accel_cstr}')
        accelator_cstr = accel_file.accel_cstr
        map_cstr = Constraint(num_pe=opt.num_pe)
        translate_to_actual_cstr(accelator_cstr, map_cstr)

    if  opt.mapping_cstr:
        mapping_file = importlib.import_module(f'data.mapping_cstr.{opt.mapping_cstr}')
        mapping_cstr = mapping_file.mapping_cstr
        map_cstr = Constraint(num_pe=opt.num_pe) if not map_cstr else map_cstr
        put_into_actual_cstr(mapping_cstr, map_cstr)

    if  opt.costmodel_cstr:
        mapping_file = importlib.import_module(f'data.mapping_cstr.advanced_cstr.costmodel_cstr.{opt.costmodel_cstr}')
        costmodel_cstr = mapping_file.mapping_cstr
        map_cstr = Constraint(num_pe=opt.num_pe) if not map_cstr else map_cstr
        put_into_actual_cstr(costmodel_cstr, map_cstr)

    if check_tpu(opt.accel_cstr, opt.mapping_cstr):
        model_defs = translate_to_gemm(model_defs)

    try:
        train_model(model_defs, input_arg=opt, map_cstr=map_cstr, chkpt_file=chkpt_file)

    finally:
        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)
