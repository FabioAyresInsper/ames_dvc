import pathlib
import pickle

import numpy as np
import pandas as pd


def main():
    DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'

    processed_file_path = DATA_DIR / 'processed' / 'ames_with_correct_types.pkl'

    with open(processed_file_path, 'rb') as file:
        (
            data,
            continuous_variables,
            discrete_variables,
            ordinal_variables,
            categorical_variables,
        ) = pickle.load(file)

    selection = ~(data['MS.Zoning'].isin(['A (agr)', 'C (all)', 'I (all)']))

    data = data[selection]

    data['MS.Zoning'] = data['MS.Zoning'].cat.remove_unused_categories()

    processed_data = data.copy()

    def remap_categories(
        series: pd.Series,
        old_categories: tuple[str],
        new_category: str,
    ) -> pd.Series:
        # Add the new category to the list of valid categories.
        series = series.cat.add_categories(new_category)

        # Set all items of the old categories as the new category.
        remapped_items = series.isin(old_categories)
        series.loc[remapped_items] = new_category

        # Clean up the list of categories, the old categories no longer exist.
        series = series.cat.remove_unused_categories()

        return series

    processed_data['Sale.Type'] = remap_categories(
        series=processed_data['Sale.Type'],
        old_categories=('WD ', 'CWD', 'VWD'),
        new_category='GroupedWD',
    )

    processed_data['Sale.Type'] = remap_categories(
        series=processed_data['Sale.Type'],
        old_categories=('COD', 'ConLI', 'Con', 'ConLD', 'Oth', 'ConLw'),
        new_category='Other',
    )

    data = processed_data

    data = data.drop(columns='Street')

    processed_data = data.copy()

    for col in ('Condition.1', 'Condition.2'):
        processed_data[col] = remap_categories(
            series=processed_data[col],
            old_categories=('RRAn', 'RRAe', 'RRNn', 'RRNe'),
            new_category='Railroad',
        )
        processed_data[col] = remap_categories(
            series=processed_data[col],
            old_categories=('Feedr', 'Artery'),
            new_category='Roads',
        )
        processed_data[col] = remap_categories(
            series=processed_data[col],
            old_categories=('PosA', 'PosN'),
            new_category='Positive',
        )

    processed_data['Condition'] = pd.Series(
        index=processed_data.index,
        dtype=pd.CategoricalDtype(categories=(
            'Norm',
            'Railroad',
            'Roads',
            'Positive',
            'RoadsAndRailroad',
        )),
    )

    norm_items = processed_data['Condition.1'] == 'Norm'
    processed_data['Condition'][norm_items] = 'Norm'

    railroad_items = \
        (processed_data['Condition.1'] == 'Railroad') \
        & (processed_data['Condition.2'] == 'Norm')
    processed_data['Condition'][railroad_items] = 'Railroad'

    roads_items = \
        (processed_data['Condition.1'] == 'Roads') \
        & (processed_data['Condition.2'] != 'Railroad')
    processed_data['Condition'][roads_items] = 'Roads'

    positive_items = processed_data['Condition.1'] == 'Positive'
    processed_data['Condition'][positive_items] = 'Positive'

    roads_and_railroad_items = \
        ( \
            (processed_data['Condition.1'] == 'Railroad') \
            & (processed_data['Condition.2'] == 'Roads')
        ) \
        | ( \
            (processed_data['Condition.1'] == 'Roads') \
            & (processed_data['Condition.2'] == 'Railroad') \
        )
    processed_data['Condition'][roads_and_railroad_items] = 'RoadsAndRailroad'

    processed_data = processed_data.drop(
        columns=['Condition.1', 'Condition.2'])

    data = processed_data

    data['HasShed'] = data['Misc.Feature'] == 'Shed'
    data = data.drop(columns='Misc.Feature')

    data['HasAlley'] = ~data['Alley'].isna()
    data = data.drop(columns='Alley')

    data['Exterior.2nd'] = remap_categories(
        series=data['Exterior.2nd'],
        old_categories=('Brk Cmn', ),
        new_category='BrkComm',
    )
    data['Exterior.2nd'] = remap_categories(
        series=data['Exterior.2nd'],
        old_categories=('CmentBd', ),
        new_category='CemntBd',
    )
    data['Exterior.2nd'] = remap_categories(
        series=data['Exterior.2nd'],
        old_categories=('Wd Shng', ),
        new_category='WdShing',
    )

    for col in ('Exterior.1st', 'Exterior.2nd'):
        categories = data[col].cat.categories
        data[col] = data[col].cat.reorder_categories(sorted(categories))

    processed_data = data.copy()

    mat_count = processed_data['Exterior.1st'].value_counts()

    rare_materials = list(mat_count[mat_count < 40].index)
    processed_data['Exterior'] = remap_categories(
        series=processed_data['Exterior.1st'],
        old_categories=rare_materials,
        new_category='Other',
    )
    processed_data = processed_data.drop(
        columns=['Exterior.1st', 'Exterior.2nd'])

    data = processed_data

    data = data.drop(columns='Heating')

    data = data.drop(columns='Roof.Matl')

    data['Roof.Style'] = remap_categories(
        series=data['Roof.Style'],
        old_categories=[
            'Flat',
            'Gambrel',
            'Mansard',
            'Shed',
        ],
        new_category='Other',
    )

    data['Mas.Vnr.Type'] = remap_categories(
        series=data['Mas.Vnr.Type'],
        old_categories=[
            'BrkCmn',
            'CBlock',
        ],
        new_category='Other',
    )

    data['Mas.Vnr.Type'] = data['Mas.Vnr.Type'].cat.add_categories('None')
    data['Mas.Vnr.Type'][data['Mas.Vnr.Type'].isna()] = 'None'

    data['MS.SubClass'] = remap_categories(
        series=data['MS.SubClass'],
        old_categories=[75, 45, 180, 40, 150],
        new_category='Other',
    )

    data['Foundation'] = remap_categories(
        series=data['Foundation'],
        old_categories=['Slab', 'Stone', 'Wood'],
        new_category='Other',
    )

    selection = ~data['Neighborhood'].isin([
        'Blueste',
        'Greens',
        'GrnHill',
        'Landmrk',
    ])
    data = data[selection]

    data['Neighborhood'] = data['Neighborhood'].cat.remove_unused_categories()

    data['Garage.Type'] = data['Garage.Type'].cat.add_categories(['NoGarage'])
    data['Garage.Type'][data['Garage.Type'].isna()] = 'NoGarage'

    all_categorical = data.select_dtypes('category').columns

    new_categorical_variables = [ \
        col for col in all_categorical \
        if not col in ordinal_variables \
    ]

    data = data.drop(columns='Utilities')

    data = data.drop(columns='Pool.QC')

    old_categories = list(data['Fence'].cat.categories)
    new_categories = old_categories + ['NoFence']

    data['Fence'] = data['Fence'].cat.set_categories(new_categories)
    data['Fence'][data['Fence'].isna()] = 'NoFence'

    data = data.drop(columns='Fireplace.Qu')

    data = data.drop(columns=['Garage.Cond', 'Garage.Qual'])

    data['Garage.Finish'] = data['Garage.Finish'] \
        .cat \
        .as_unordered() \
        .cat \
        .add_categories(['NoGarage'])
    data['Garage.Finish'][data['Garage.Finish'].isna()] = 'NoGarage'

    data['Electrical'][data['Electrical'].isna()] = 'SBrkr'

    data['Bsmt.Exposure'][data['Bsmt.Exposure'].isna()] = 'NA'
    data['Bsmt.Exposure'] = data['Bsmt.Exposure'] \
        .cat \
        .as_unordered() \
        .cat \
        .remove_unused_categories()

    for col in ('Bsmt.Qual', 'Bsmt.Cond', 'BsmtFin.Type.1', 'BsmtFin.Type.2'):
        data[col] = data[col].cat.add_categories(['NA'])
        data[col][data[col].isna()] = 'NA'
        data[col] = data[col] \
            .cat \
            .as_unordered() \
            .cat \
            .remove_unused_categories()

    data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Po'] = 'Fa'
    data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Ex'] = 'Gd'
    data['Bsmt.Cond'] = data['Bsmt.Cond'].cat.remove_unused_categories()

    data['SalePrice'] = data['SalePrice'].apply(np.log10)

    data['Lot.Frontage'] = data['Lot.Frontage'].fillna(
        data['Lot.Frontage'].median())

    data['Garage.Yr.Blt'].describe()

    garage_age = data['Yr.Sold'] - data['Garage.Yr.Blt']

    garage_age[garage_age < 0.0] = 0.0

    data = data.drop(columns='Garage.Yr.Blt')
    data['Garage.Age'] = garage_age

    data['Garage.Age'] = data['Garage.Age'].fillna(data['Garage.Age'].median())

    remod_age = data['Yr.Sold'] - data['Year.Remod.Add']
    remod_age[remod_age < 0.0] = 0.0

    house_age = data['Yr.Sold'] - data['Year.Built']
    house_age[house_age < 0.0] = 0.0

    data = data.drop(columns=['Year.Remod.Add', 'Year.Built'])
    data['Remod.Age'] = remod_age
    data['House.Age'] = house_age

    data.loc[data['Mas.Vnr.Area'].isna(), 'Mas.Vnr.Area'] = 0.0

    for col in data.select_dtypes('category').columns:
        data[col] = data[col].cat.remove_unused_categories()

    data = data.dropna()

    clean_data_path = DATA_DIR / 'processed' / 'ames_clean.pkl'

    with open(clean_data_path, 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    main()
