import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# Read the datasets
elements_df = pd.read_csv('elements.csv')
ksp_df = pd.read_csv('ksp.csv')
radius_df = pd.read_csv('radius.csv')

# Create a dictionary for quick lookup of element properties
element_properties = elements_df.set_index('Symbol').T.to_dict()
print(element_properties)
ions_set = radius_df.set_index('Ion').T.to_dict()
ions_set = list(reversed(list(dict(sorted(ions_set.items(), key=lambda item: len(item[0]))).keys())))
print(ions_set)

# Function to parse chemical formulas and return a dictionary of element counts
def get_elem_counts(formula):
    # Regex to match element symbols and their counts
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)
    element_counts = {}
    for (element, count) in matches:
        count = int(count) if count else 1
        if element in element_counts:
            element_counts[element] += count
        else:
            element_counts[element] = count
    return element_counts

# def get_charges(cation, anion):



def get_mass(formula):
    element_counts = get_elem_counts(formula)
    mass = 0
    for element in element_counts.keys():
        mass += element_properties[element]['AtomicMass'] * element_counts[element]
    return mass

def parse_formula(formula):
    elements_present = {}
    print(formula)
    formula = formula.replace("H2O", "", 1)
    for ion in ions_set:
        if re.findall(ion, formula):
            print(ion)
            freq_index = formula.index(ion) + len(ion)
            freq = 1
            if freq_index < len(formula):
                if formula[freq_index] == ')':
                    freq_index += 1
                if freq_index < len(formula) and formula[freq_index].isnumeric():
                    freq = int(formula[freq_index])
            formula = formula.replace(ion, "", 1)
            elements_present.update({ion : {"Frequency" : freq}})
    return elements_present


def replace_first(pattern, replacement, string):
    index = string.find(pattern)
    if index != -1:
        return string[:index] + replacement + string[index + len(pattern):]
    return string

# Function to convert scientific notation strings to float
def parse_scientific_notation(value):
    try:
        # Replace Unicode multiplication sign and Unicode minus sign with standard equivalents
        clean_value = value.replace('×', 'e').replace('−', '-').replace(' ', '').replace('10', '', 1)
        return float(clean_value)
    except ValueError:
        return np.nan


# Initialize list to store the new dataset
new_dataset = []

# Process each compound in the ksp dataset
for index, row in ksp_df.iterrows():
    compound_name = row['Compound Name']
    compound_formula = row['Compound Formula']
    ksp_value_str = row['Ksp Value']

    # Parse the Ksp value
    ksp_value = parse_scientific_notation(ksp_value_str)
    if pd.isna(ksp_value):
        print(f"Failed to parse Ksp value for {compound_name}: {ksp_value_str}")

    # Parse the compound formula to get element counts
    elements_present = parse_formula(compound_formula)

    new_dataset.append(elements_present)
# print(len(new_dataset))
new_dataset = [data for data in new_dataset if len(data) == 2]
print(new_dataset)
for data in new_dataset:
    for ion in data:
        data[ion].update({"Mass" : get_mass(ion)})


print(new_dataset)
#
#     # Create a dictionary to store the combined data for this compound
#     compound_data = {
#         'Compound Name': compound_name,
#         'Compound Formula': compound_formula,
#         'Ksp Value': ksp_value,
#     }
#
#     # Initialize properties aggregation
#     total_properties = {
#         'AtomicNumber': 0, 'AtomicMass': 0, 'Electronegativity': 0, 'AtomicRadius': 0,
#         'IonizationEnergy': 0, 'ElectronAffinity': 0, 'MeltingPoint': 0, 'BoilingPoint': 0, 'Density': 0,
#         'ElementCount': 0  # to normalize aggregated properties
#     }
#
#     # Aggregate element properties
#     for element, count in element_counts.items():
#         if element in element_properties:
#             prop = element_properties[element]
#             total_properties['AtomicNumber'] += prop['AtomicNumber'] * count
#             total_properties['AtomicMass'] += prop['AtomicMass'] * count
#             total_properties['Electronegativity'] += (prop['Electronegativity'] if pd.notna(
#                 prop['Electronegativity']) else 0) * count
#             total_properties['AtomicRadius'] += (prop['AtomicRadius'] if pd.notna(prop['AtomicRadius']) else 0) * count
#             total_properties['IonizationEnergy'] += (prop['IonizationEnergy'] if pd.notna(
#                 prop['IonizationEnergy']) else 0) * count
#             total_properties['ElectronAffinity'] += (prop['ElectronAffinity'] if pd.notna(
#                 prop['ElectronAffinity']) else 0) * count
#             total_properties['MeltingPoint'] += (prop['MeltingPoint'] if pd.notna(prop['MeltingPoint']) else 0) * count
#             total_properties['BoilingPoint'] += (prop['BoilingPoint'] if pd.notna(prop['BoilingPoint']) else 0) * count
#             total_properties['Density'] += (prop['Density'] if pd.notna(prop['Density']) else 0) * count
#             total_properties['ElementCount'] += count
#
#     # Normalize the aggregated properties by the total count of elements
#     for key in total_properties:
#         if key != 'ElementCount' and total_properties['ElementCount'] > 0:
#             total_properties[key] /= total_properties['ElementCount']
#
#     # Combine with the compound data
#     compound_data.update(total_properties)
#
#     # Append the compound data to the new dataset
#     new_dataset.append(compound_data)
#
# # Convert the new dataset to a DataFrame
# new_df = pd.DataFrame(new_dataset)
#
# # Drop the columns that are not needed for the ML model
# new_df = new_df.drop(columns=['Compound Name', 'Compound Formula'])
#
# # Ensure all numeric columns are float and replace NaNs with 0
# new_df = new_df.fillna(0)
# for col in new_df.columns:
#     new_df[col] = new_df[col].astype(float)
#
# # Normalize the numeric data between 0 and 1
# scaler = MinMaxScaler()
# normalized_df = pd.DataFrame(scaler.fit_transform(new_df), columns=new_df.columns)
#
# # Save the normalized dataset to a CSV file
# normalized_df.to_csv('normalized_combined_dataset.csv', index=False)
