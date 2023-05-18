categorical_bool_columns = [
    'rbc',
    'pc',
    'pcc',
    'ba',
    'htn',
    'dm',
    'cad',
    'appet',
    'pe',
    'ane',
    'class'
]
categorical_num_columns = ['sg', 'al', 'su']
categorical_columns = categorical_bool_columns + categorical_num_columns
numerical_columns = [
    'age',
    'bp',
    'bgr',
    'bu',
    'sc',
    'sod',
    'pot',
    'hemo',
    'pcv',
    'wbcc',
    'rbcc',
]
all_columns = categorical_columns + numerical_columns


map_bool_to_val = {
    'rbc': {
        True: 'normal',
        False: 'abnormal'
    },
    'pc': {
        True: 'normal',
        False: 'abnormal'
    },
    'pcc': {
        True: 'present',
        False: 'notpresent'
    },
    'ba': {
        True: 'present',
        False: 'notpresent'
    },
    'htn': {
        True: 'yes',
        False: 'no'
    },
    'dm': {
        True: 'yes',
        False: 'no'
    },
    'cad': {
        True: 'yes',

        False: 'no'
    },
    'appet': {
        True: 'good',
        False: 'poor'
    },
    'pe': {
        True: 'yes',
        False: 'no'
    },
    'ane': {
        True: 'yes',
        False: 'no'
    },
    'class': {
        True: 'ckd',
        False: 'notckd'
    },
}
