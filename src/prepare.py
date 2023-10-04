import pickle
from pathlib import Path

import pandas as pd
import requests


def main():
    DATA_DIR = Path(__file__).parents[1] / 'data'

    raw_data_file_path = DATA_DIR / 'raw' / 'ames.csv'
    data = pd.read_csv(raw_data_file_path)

    ignore_variables = [
        'Order',
        'PID',
    ]

    continuous_variables = [
        'Lot.Frontage',
        'Lot.Area',
        'Mas.Vnr.Area',
        'BsmtFin.SF.1',
        'BsmtFin.SF.2',
        'Bsmt.Unf.SF',
        'Total.Bsmt.SF',
        'X1st.Flr.SF',
        'X2nd.Flr.SF',
        'Low.Qual.Fin.SF',
        'Gr.Liv.Area',
        'Garage.Area',
        'Wood.Deck.SF',
        'Open.Porch.SF',
        'Enclosed.Porch',
        'X3Ssn.Porch',
        'Screen.Porch',
        'Pool.Area',
        'Misc.Val',
        'SalePrice',
    ]

    discrete_variables = [
        'Year.Built',
        'Year.Remod.Add',
        'Bsmt.Full.Bath',
        'Bsmt.Half.Bath',
        'Full.Bath',
        'Half.Bath',
        'Bedroom.AbvGr',
        'Kitchen.AbvGr',
        'TotRms.AbvGrd',
        'Fireplaces',
        'Garage.Yr.Blt',
        'Garage.Cars',
        'Mo.Sold',
        'Yr.Sold',
    ]

    ordinal_variables = [
        'Lot.Shape',
        'Utilities',
        'Land.Slope',
        'Overall.Qual',
        'Overall.Cond',
        'Exter.Qual',
        'Exter.Cond',
        'Bsmt.Qual',
        'Bsmt.Cond',
        'Bsmt.Exposure',
        'BsmtFin.Type.1',
        'BsmtFin.Type.2',
        'Heating.QC',
        'Electrical',
        'Kitchen.Qual',
        'Functional',
        'Fireplace.Qu',
        'Garage.Finish',
        'Garage.Qual',
        'Garage.Cond',
        'Paved.Drive',
        'Pool.QC',
        'Fence',
    ]

    categorical_variables = [
        'MS.SubClass',
        'MS.Zoning',
        'Street',
        'Alley',
        'Land.Contour',
        'Lot.Config',
        'Neighborhood',
        'Condition.1',
        'Condition.2',
        'Bldg.Type',
        'House.Style',
        'Roof.Style',
        'Roof.Matl',
        'Exterior.1st',
        'Exterior.2nd',
        'Mas.Vnr.Type',
        'Foundation',
        'Heating',
        'Central.Air',
        'Garage.Type',
        'Misc.Feature',
        'Sale.Type',
        'Sale.Condition',
    ]

    data.drop(columns=ignore_variables, inplace=True)

    for col in continuous_variables:
        data[col] = data[col].astype('float64')

    for col in categorical_variables:
        data[col] = data[col].astype('category')

    for col in discrete_variables:
        data[col] = data[col].astype('float64')

    category_orderings = {
        'Lot.Shape': [
            'Reg',
            'IR1',
            'IR2',
            'IR3',
        ],
        'Utilities': [
            'AllPub',
            'NoSewr',
            'NoSeWa',
            'ELO',
        ],
        'Land.Slope': [
            'Gtl',
            'Mod',
            'Sev',
        ],
        'Overall.Qual': [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
        'Overall.Cond': [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
        'Exter.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Exter.Cond': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Bsmt.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Bsmt.Cond': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Bsmt.Exposure': [
            'Gd',
            'Av',
            'Mn',
            'No',
            'NA',
        ],
        'BsmtFin.Type.1': [
            'GLQ',
            'ALQ',
            'BLQ',
            'Rec',
            'LwQ',
            'Unf',
        ],
        'BsmtFin.Type.2': [
            'GLQ',
            'ALQ',
            'BLQ',
            'Rec',
            'LwQ',
            'Unf',
        ],
        'Heating.QC': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Electrical': [
            'SBrkr',
            'FuseA',
            'FuseF',
            'FuseP',
            'Mix',
        ],
        'Kitchen.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Functional': [
            'Typ',
            'Min1',
            'Min2',
            'Mod',
            'Maj1',
            'Maj2',
            'Sev',
            'Sal',
        ],
        'Fireplace.Qu': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Garage.Finish': [
            'Fin',
            'RFn',
            'Unf',
        ],
        'Garage.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Garage.Cond': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Paved.Drive': [
            'Y',
            'P',
            'N',
        ],
        'Pool.QC': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
        ],
        'Fence': [
            'GdPrv',
            'MnPrv',
            'GdWo',
            'MnWw',
        ],
    }

    for col, orderings in category_orderings.items():
        data[col] = data[col] \
            .astype('category') \
            .cat \
            .set_categories(orderings, ordered=True)

    processed_dir = DATA_DIR / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_file_path = processed_dir / 'ames_with_correct_types.pkl'

    with open(processed_file_path, 'wb') as file:
        pickle.dump(
            [
                data,
                continuous_variables,
                discrete_variables,
                ordinal_variables,
                categorical_variables,
            ],
            file,
        )


if __name__ == '__main__':
    main()
