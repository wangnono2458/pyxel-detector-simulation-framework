#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyxel.detectors import CCD, CCDGeometry, Characteristics, Detector, Environment
from pyxel.models.charge_generation import cosmix


@pytest.fixture
def ccd_8x8() -> CCD:
    detector = CCD(
        geometry=CCDGeometry(
            row=8,
            col=8,
            pixel_horz_size=10.0,
            pixel_vert_size=10.0,
            total_thickness=40.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector.set_readout(times=[1.0, 5.0, 7.0], non_destructive=False)

    return detector


@pytest.mark.parametrize(
    "extra_params",
    [
        pytest.param({}, id="without extra parameters"),
        pytest.param(
            {"incident_angles": None, "starting_position": None},
            id="with empties 'incident_angles' and 'starting_positions'",
        ),
        pytest.param(
            {
                "stepsize": [
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 40.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_40um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 50.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_50um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 60.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_60um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 70.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_70um_Si_10k.ascii",
                    },
                    {
                        "type": "proton",
                        "energy": 100.0,
                        "thickness": 100.0,
                        "filename": "{folder}/data/stepsize_proton_100MeV_100um_Si_10k.ascii",
                    },
                ]
            },
            id="with 'stepsize'",
        ),
    ],
)
def test_cosmix_stepsize(extra_params, ccd_8x8: CCD, request: pytest.FixtureRequest):
    detector: Detector = ccd_8x8
    assert isinstance(extra_params, dict)

    charge_2d = np.array(
        [
            [14323.0, 13206.0, 13087.0, 13152.0, 13165.0, 13138.0, 13106.0, 13162.0],
            [13284.0, 11870.0, 11779.0, 11877.0, 11887.0, 11853.0, 11797.0, 11797.0],
            [13231.0, 11856.0, 11788.0, 11875.0, 11892.0, 11859.0, 11789.0, 11715.0],
            [13331.0, 11971.0, 11861.0, 11946.0, 11975.0, 11914.0, 11842.0, 11776.0],
            [13387.0, 12004.0, 11868.0, 11969.0, 11963.0, 11816.0, 11740.0, 11765.0],
            [13416.0, 12022.0, 11887.0, 11977.0, 11894.0, 11659.0, 11592.0, 11727.0],
            [13404.0, 12034.0, 11930.0, 11988.0, 11829.0, 11558.0, 11508.0, 11686.0],
            [13358.0, 12026.0, 11948.0, 11952.0, 11755.0, 11513.0, 11443.0, 11536.0],
        ]
    )
    detector.charge.add_charge_array(charge_2d)

    current_folder: Path = request.path.parent

    if "stepsize" in extra_params:
        new_stepsizes = [
            {
                key: value if key != "filename" else value.format(folder=current_folder)
                for key, value in element.items()
            }
            for element in extra_params["stepsize"]
        ]

        extra_params["stepsize"] = new_stepsizes

    cosmix(
        detector=detector,
        simulation_mode="cosmic_ray",
        running_mode="stepsize",
        particle_type="proton",
        initial_energy=100.0,
        particles_per_second=100.0,
        spectrum_file=str(
            current_folder / "data/proton_L2_solarMax_11mm_Shielding.txt"
        ),
        seed=1234,
        progressbar=False,
        **extra_params,
    )

    new_charges_df = detector.charge.frame

    assert isinstance(new_charges_df, pd.DataFrame)
    assert len(new_charges_df) == 4044

    exp_head_df = pd.DataFrame(
        {
            "charge": [-1, -1, -1, -1, -1],
            "number": [14323.0, 13206.0, 13087.0, 13152.0, 13165.0],
            "init_energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "init_pos_ver": [5.0, 5.0, 5.0, 5.0, 5.0],
            "init_pos_hor": [5.0, 15.0, 25.0, 35.0, 45.0],
            "init_pos_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "position_ver": [5.0, 5.0, 5.0, 5.0, 5.0],
            "position_hor": [5.0, 15.0, 25.0, 35.0, 45.0],
            "position_z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_ver": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_hor": [0.0, 0.0, 0.0, 0.0, 0.0],
            "velocity_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(new_charges_df.head(), exp_head_df)

    exp_tail_df = pd.DataFrame(
        {
            "charge": {4039: -1, 4040: -1, 4041: -1, 4042: -1, 4043: -1},
            "number": {4039: 1.0, 4040: 1.0, 4041: 1.0, 4042: 1.0, 4043: 1.0},
            "init_energy": {
                4039: 1000.0,
                4040: 1000.0,
                4041: 1000.0,
                4042: 1000.0,
                4043: 1000.0,
            },
            "energy": {
                4039: 1000.0,
                4040: 1000.0,
                4041: 1000.0,
                4042: 1000.0,
                4043: 1000.0,
            },
            "init_pos_ver": {
                4039: 24.146376064873532,
                4040: 22.512833964453264,
                4041: 22.422229542748024,
                4042: 19.792877370333045,
                4043: 18.645005809003585,
            },
            "init_pos_hor": {
                4039: 21.000567801654356,
                4040: 19.48662185641866,
                4041: 19.402650836127368,
                4042: 16.965800840648107,
                4043: 15.901968145133862,
            },
            "init_pos_z": {
                4039: -2.0456102252095296,
                4040: -1.5831256464107024,
                4041: -1.5574739366697175,
                4042: -0.813057963760132,
                4043: -0.4880752801672042,
            },
            "position_ver": {
                4039: 24.146376064873532,
                4040: 22.512833964453264,
                4041: 22.422229542748024,
                4042: 19.792877370333045,
                4043: 18.645005809003585,
            },
            "position_hor": {
                4039: 21.000567801654356,
                4040: 19.48662185641866,
                4041: 19.402650836127368,
                4042: 16.965800840648107,
                4043: 15.901968145133862,
            },
            "position_z": {
                4039: -2.0456102252095296,
                4040: -1.5831256464107024,
                4041: -1.5574739366697175,
                4042: -0.813057963760132,
                4043: -0.4880752801672042,
            },
            "velocity_ver": {4039: 0.0, 4040: 0.0, 4041: 0.0, 4042: 0.0, 4043: 0.0},
            "velocity_hor": {4039: 0.0, 4040: 0.0, 4041: 0.0, 4042: 0.0, 4043: 0.0},
            "velocity_z": {4039: 0.0, 4040: 0.0, 4041: 0.0, 4042: 0.0, 4043: 0.0},
        }
    )
    pd.testing.assert_frame_equal(new_charges_df.tail(), exp_tail_df, check_exact=False)

    new_charges_2d = detector.charge.array
    assert isinstance(new_charges_2d, np.ndarray)

    exp_charges_2d = np.array(
        [
            [17769.0, 14661.0, 14295.0, 15721.0, 20829.0, 19341.0, 28128.0, 22383.0],
            [15562.0, 15465.0, 22929.0, 17084.0, 21102.0, 20532.0, 25860.0, 13457.0],
            [16760.0, 13171.0, 16895.0, 20608.0, 13701.0, 21190.0, 15862.0, 17818.0],
            [15187.0, 14118.0, 15079.0, 22020.0, 12045.0, 14818.0, 16927.0, 14118.0],
            [24062.0, 22221.0, 26730.0, 16113.0, 18177.0, 15021.0, 23801.0, 13435.0],
            [14091.0, 14230.0, 15044.0, 15197.0, 20576.0, 29355.0, 13355.0, 14583.0],
            [13818.0, 14278.0, 14461.0, 13441.0, 19379.0, 18477.0, 13639.0, 13211.0],
            [15011.0, 12580.0, 15847.0, 13072.0, 16862.0, 11969.0, 16321.0, 12402.0],
        ]
    )
    np.testing.assert_almost_equal(new_charges_2d, exp_charges_2d)
