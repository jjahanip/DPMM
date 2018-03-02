function ch = get_channels(dataset)

if strcmp(dataset, 'LiVPa')
    ch  = {...
        'GAD67'       ,'ARBc_#4_Li+VPA_37C_4110_C1_IlluminationCorrected_stitched.tif'              , [0 0 1];
        'Parvalbumin' ,'ARBc_#4_Li+VPA_37C_4110_C2_IlluminationCorrected_stitched.tif'              , [0 1 1];
        'APC'         ,'ARBc_#4_Li+VPA_37C_4110_C4_IlluminationCorrected_stitched.tif'              , [0 1 .5];
        'RECA1'       ,'ARBc_#4_Li+VPA_37C_4110_C5_IlluminationCorrected_stitched.tif'              , [0 1 0];
        'S100'        ,'ARBc_#4_Li+VPA_37C_4110_C6_IlluminationCorrected_stitched.tif'              , [1 1 0];
        'NeuN'        ,'ARBc_#4_Li+VPA_37C_4110_C7_IlluminationCorrected_stitched.tif'              , [1 .5 0];
        'IBA1'        ,'ARBc_#4_Li+VPA_37C_4110_C8_IlluminationCorrected_stitched.tif'              , [1 0 0];
        'DAPI'        ,'ARBc_#4_Li+VPA_37C_4110_C10_IlluminationCorrected_stitched.tif'             , [1 1 1];
        'TubulinBeta3','ARBc_#4_Li+VPA_37C_4111_C1_IlluminationCorrected_stitched_registered.tif'   , [1 0 .5];
        'MAP2'        ,'ARBc_#4_Li+VPA_37C_4111_C2_IlluminationCorrected_stitched_registered.tif'   , [1 0 1];
        'PLP'         ,'ARBc_#4_Li+VPA_37C_4111_C5_IlluminationCorrected_stitched_registered.tif'   , [.5 0 1];
        'GFAP'        ,'ARBc_#4_Li+VPA_37C_4111_C8_IlluminationCorrected_stitched_registered.tif'   , [0 .6 1];
        'PCNA'        ,'ARBc_#4_Li+VPA_37C_4112_C5_IlluminationCorrected_stitched_registered.tif'   , [0 1 1];
        'CC3'         ,'ARBc_#4_Li+VPA_37C_4112_C6_IlluminationCorrected_stitched_registered.tif'   , [0 1 .5];
        'NFH'         ,'ARBc_#4_Li+VPA_37C_4112_C7_IlluminationCorrected_stitched_registered.tif'   , [0 1 0];
        'Claretinin'  ,'ARBc_#4_Li+VPA_37C_4113_C4_IlluminationCorrected_stitched_registered.tif'   , [1 1 0];
        'SynaptoPhys' ,'ARBc_#4_Li+VPA_37C_4113_C5_IlluminationCorrected_stitched_registered.tif'   , [1 .5 0];
        'GLAST'       ,'ARBc_#4_Li+VPA_37C_4113_C6_IlluminationCorrected_stitched_registered.tif'   , [1 0 0];
        'MBP'         ,'ARBc_#4_Li+VPA_37C_4113_C7_IlluminationCorrected_stitched_registered.tif'   , [1 0 .5];
        'TomatoLectin','ARBc_#4_Li+VPA_37C_4113_C8_IlluminationCorrected_stitched_registered.tif'   , [1 0 1];};
elseif strcmp(dataset, 'sham')
    ch = {...
        'GAD67'       ,'ARBc_#29_Sham_1C_4110_C1_IlluminationCorrected_stitched.tif'                , [0 0 1];
        'Parvalbumin' ,'ARBc_#29_Sham_1C_4110_C2_IlluminationCorrected_stitched.tif'                , [0 1 1];
        'APC'         ,'ARBc_#29_Sham_1C_4110_C4_IlluminationCorrected_stitched.tif'                , [0 1 .5];
        'RECA1'       ,'ARBc_#29_Sham_1C_4110_C5_IlluminationCorrected_stitched.tif'                , [0 1 0];
        'S100'        ,'ARBc_#29_Sham_1C_4110_C6_IlluminationCorrected_stitched.tif'                , [1 1 0];
        'NeuN'        ,'ARBc_#29_Sham_1C_4110_C7_IlluminationCorrected_stitched.tif'                , [1 .5 0];
        'IBA1'        ,'ARBc_#29_Sham_1C_4110_C8_IlluminationCorrected_stitched.tif'                , [1 0 0];
        'DAPI'        ,'ARBc_#29_Sham_1C_4110_C10_IlluminationCorrected_stitched.tif'               , [1 1 1];
        'TubulinBeta3','ARBc_#29_Sham_1C_4111_C1_IlluminationCorrected_stitched_registered.tif'     , [1 0 .5];
        'MAP2'        ,'ARBc_#29_Sham_1C_4111_C2_IlluminationCorrected_stitched_registered.tif'     , [1 0 1];
        'PLP'         ,'ARBc_#29_Sham_1C_4111_C5_IlluminationCorrected_stitched_registered.tif'     , [.5 0 1];
        'GFAP'        ,'ARBc_#29_Sham_1C_4111_C8_IlluminationCorrected_stitched_registered.tif'     , [0 1 1];
        'PCNA'        ,'ARBc_#29_Sham_1C_4112_C5_IlluminationCorrected_stitched_registered.tif'     , [0 1 1];
        'CC3'         ,'ARBc_#29_Sham_1C_4112_C6_IlluminationCorrected_stitched_registered.tif'     , [0 1 .5];
        'NFH'         ,'ARBc_#29_Sham_1C_4112_C7_IlluminationCorrected_stitched_registered.tif'     , [0 1 0];
        'Claretinin'  ,'ARBc_#29_Sham_1C_4113_C4_IlluminationCorrected_stitched_registered.tif'     , [1 1 0];
        'SynaptoPhys' ,'ARBc_#29_Sham_1C_4113_C5_IlluminationCorrected_stitched_registered.tif'     , [1 .5 0];
        'GLAST'       ,'ARBc_#29_Sham_1C_4113_C6_IlluminationCorrected_stitched_registered.tif'     , [1 0 0];
        'MBP'         ,'ARBc_#29_Sham_1C_4113_C7_IlluminationCorrected_stitched_registered.tif'     , [1 0 .5];
        'TomatoLectin','ARBc_#29_Sham_1C_4113_C8_IlluminationCorrected_stitched_registered.tif'     , [1 0 1];};
elseif strcmp(dataset, 'vehicle')
    ch = {...
        'GAD67'       ,'ARBc_FPI#6_Vehicle_20C_4110_C1_IlluminationCorrected_stitched.tif'           , [0 0 1];
        'Parvalbumin' ,'ARBc_FPI#6_Vehicle_20C_4110_C2_IlluminationCorrected_stitched.tif'           , [0 1 1];
        'APC'         ,'ARBc_FPI#6_Vehicle_20C_4110_C4_IlluminationCorrected_stitched.tif'           , [0 1 .5];
        'RECA1'       ,'ARBc_FPI#6_Vehicle_20C_4110_C5_IlluminationCorrected_stitched.tif'           , [0 1 0];
        'S100'        ,'ARBc_FPI#6_Vehicle_20C_4110_C6_IlluminationCorrected_stitched.tif'           , [1 1 0];
        'NeuN'        ,'ARBc_FPI#6_Vehicle_20C_4110_C7_IlluminationCorrected_stitched.tif'           , [1 .5 0];
        'IBA1'        ,'ARBc_FPI#6_Vehicle_20C_4110_C8_IlluminationCorrected_stitched.tif'           , [1 0 0];
        'DAPI'        ,'ARBc_FPI#6_Vehicle_20C_4110_C10_IlluminationCorrected_stitched.tif'          , [1 1 1];
        'TubulinBeta3','ARBc_FPI#6_Vehicle_20C_4111_C1_IlluminationCorrected_stitched_registered.tif', [1 0 .5];
        'MAP2'        ,'ARBc_FPI#6_Vehicle_20C_4111_C2_IlluminationCorrected_stitched_registered.tif', [1 0 1];
        'PLP'         ,'ARBc_FPI#6_Vehicle_20C_4111_C5_IlluminationCorrected_stitched_registered.tif', [.5 0 1];
        'GFAP'        ,'ARBc_FPI#6_Vehicle_20C_4111_C8_IlluminationCorrected_stitched_registered.tif', [0 1 1];
        'PCNA'        ,'ARBc_FPI#6_Vehicle_20C_4112_C5_IlluminationCorrected_stitched_registered.tif', [0 1 1];
        'CC3'         ,'ARBc_FPI#6_Vehicle_20C_4112_C6_IlluminationCorrected_stitched_registered.tif', [0 1 .5];
        'NFH'         ,'ARBc_FPI#6_Vehicle_20C_4112_C7_IlluminationCorrected_stitched_registered.tif', [0 1 0];
        'Claretinin'  ,'ARBc_FPI#6_Vehicle_20C_4113_C4_IlluminationCorrected_stitched_registered.tif', [1 1 0];
        'SynaptoPhys' ,'ARBc_FPI#6_Vehicle_20C_4113_C5_IlluminationCorrected_stitched_registered.tif', [1 .5 0];
        'GLAST'       ,'ARBc_FPI#6_Vehicle_20C_4113_C6_IlluminationCorrected_stitched_registered.tif', [1 0 0];
        'MBP'         ,'ARBc_FPI#6_Vehicle_20C_4113_C7_IlluminationCorrected_stitched_registered.tif', [1 0 .5];
        'TomatoLectin','ARBc_FPI#6_Vehicle_20C_4113_C8_IlluminationCorrected_stitched_registered.tif', [1 0 1];};
elseif strcmp(dataset, 'xiaoyang_crops')
    ch = {...
        'DAPI'        ,'DAPI.tif'       , [1 1 1];
        'Histone'     ,'Histone.tif'    , [1 0 0];
        'NeuN'        ,'NeuN.tif'       , [1 .5 0];
        'Parvalbumin' ,'Parv.tif'       , [0 1 1];
        'Glut'        ,'Glut.tif'       , [1 0 1];};
end