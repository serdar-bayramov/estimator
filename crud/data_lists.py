"""
data lists for various dropdowns
"""
import streamlit as st
from typing import List
from sqlalchemy import and_, func
from sqlalchemy.orm import Session
from models import BitBasicData, CuttingStructure, BOMData, CutterAnalysis
import pandas as pd
from utils.util_functions import find_max_flow_rate, calculate_total_tfa, group_and_aggregate
from exceptions import NotFoundException
from utils.db import make_session
import time

columns_to_keep = ['bit_diameter_in', 'bit_part_id', 'csid', 'number_of_blades',
                   'type', 'shape', 'location', 'logger_pocket']


def drop_unneeded_columns(df: pd.DataFrame, columns_to_keep: List[str]) -> pd.DataFrame:
    return df[columns_to_keep]


# def query_inventory_data(db: Session, bit_size: List[float]):
#     # TO DO: QUERY DATA FROM DB
#     return

@st.cache_data
def query_inventory_data():
    df = pd.read_csv('data/bom_components.csv')
    # Split the 'child_part_no_and_hash' column on the underscore
    split_df = df['child_part_no_and_hash'].str.split('_', expand=True)
    # If the split operation produces two parts, take the first part; otherwise, take the entire value
    df['PID'] = split_df[0].where(split_df[1].notna(), df['child_part_no_and_hash'])
    
    df = df[['PID', 'desc_1', 'cost']]
    df = df.rename(columns={'desc_1': 'Description', 'cost': 'Cost (USD)'})
    return df


def query_dba_data(db: Session, bit_size: List[float]):
    """
    Query dba data
    """
    dba_data = db.query(BitBasicData).filter(BitBasicData.bit_diameter_in.in_(bit_size)).all()
    dba_df = pd.DataFrame([data.__dict__ for data in dba_data])
    return dba_df


def query_bom_data(db: Session, unique_hashes: List[str]):
    """
    Query bom data for only unique hashes that are common between dba and bom data
    """
    bom_data = db.query(BOMData).filter(
        func.split_part(BOMData.child_part_no_and_hash, '_', 2).in_(unique_hashes)).all()
    bom_df = pd.DataFrame([data.__dict__ for data in bom_data])
    return bom_df


def get_bit_sizes(db: Session):
    """
    get all bit sizes in database
    """
    bit_sizes = db.query(BitBasicData.bit_diameter_in).distinct().all()
    bit_sizes = [bit_size[0] for bit_size in bit_sizes if bit_size[0] is not None]

    if bit_sizes:
        # Sort bit sizes in descending order
        sorted_bit_sizes = sorted(bit_sizes, key=float, reverse=True)
        # Convert sorted bit sizes back to strings
        sorted_bit_sizes = [str(size) for size in sorted_bit_sizes]
        return sorted_bit_sizes
    else:
        raise NotFoundException("No bit sizes found in database")
    

@st.cache_data(ttl=3600)
def get_cached_bit_sizes():
    session = make_session(remote=True)
    try:
        bit_sizes = get_bit_sizes(db=session)
    finally:
        session.close()
    return bit_sizes


@st.cache_data(ttl=3600)
def get_cached_bom_data(unique_hashes: List[str]):
    session = make_session(remote=True)
    try:
        bom_data = query_bom_data(db=session, unique_hashes=unique_hashes)
    finally:
        session.close()
    return bom_data


@st.cache_data(ttl=3600)
def get_cached_dba_data(bit_size: List[float]):
    session = make_session(remote=True)
    try:
        dba_data = query_dba_data(db=session, bit_size=bit_size)
    finally:
        session.close()
    return dba_data


def choose_bits_from_existing_designs(dba_df, bom_df, bit_part_id_dropdown):
    cols = st.columns(3)  # Create three columns
    col_index = 0  # Initialize column index
    selected_cards = []  # List to store selected cards

    # If no bit_part_id is selected, use all part_numbers from dba_df
    if not bit_part_id_dropdown:
        bit_part_id_dropdown = dba_df['part_number'].tolist()

    for bit_part_id in bit_part_id_dropdown:
        # Filter the dataframes
        filtered_dba_df = dba_df[dba_df['part_number'] == bit_part_id]
        filtered_bom_df = bom_df[bom_df['bit_part_id'] == bit_part_id]

        # Get the required information from dba_df
        csid = filtered_dba_df['csid'].values[0]
        bit_diameter_in = filtered_dba_df['bit_diameter_in'].values[0]
        number_of_blades = filtered_dba_df['number_of_blades'].values[0]
        gauge_length_in = filtered_dba_df['gauge_length_in'].values[0]
        recon_slot = filtered_dba_df['logger_pocket'].values[0]

        # Create a list to store the information from bom_df
        bom_info = []
        for _, row in filtered_bom_df.iterrows():
            desc_1 = row['desc_1']
            c_qty = row['c_qty']
            bom_info.append({'Description': desc_1, 'Qty': c_qty})

        # Display the information in the container
        bom_info_str = "".join([f"<p style='margin: 5px 0; font-style: italic; font-size: 0.9em;'>{info['Description']}: {info['Qty']}</p>" for info in bom_info])
        with cols[col_index]:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            selected = st.checkbox(f"Add **{bit_part_id}** to cart", key=f"select_{bit_part_id}")
            if selected:
                for info in bom_info:
                    selected_cards.append({
                        'bit_part_id': bit_part_id,
                        'csid': csid,
                        'bit_diameter_in': bit_diameter_in,
                        'number_of_blades': number_of_blades,
                        'gauge_length_in': gauge_length_in,
                        'Description': info['Description'],
                        'Qty': info['Qty'],
                        'Recon slot': recon_slot
                    })

            st.markdown(
                f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: #f3f4f9; height: 400px; overflow-y: auto;">
                    <p style="margin: 5px 0; padding: 0;"><strong>{bit_part_id} | {csid}</strong></p>
                    <p style="margin: 5px 0; padding: 0; font-style: italic; font-size: 0.9em;">Bit diameter: {bit_diameter_in} in</p>
                    <p style="margin: 5px 0; padding: 0; font-style: italic; font-size: 0.9em;">Number of blades: {number_of_blades}</p>
                    <p style="margin: 5px 0; padding: 0; font-style: italic; font-size: 0.9em;">Gauge length: {gauge_length_in}</p>
                    {bom_info_str}
                    <p style="margin: 5px 0; padding: 0; font-style: italic; font-size: 0.9em;">Recon slot: {recon_slot}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Update the column index
        col_index = (col_index + 1) % 3
    return selected_cards


def calculate_total_cost(df):
    # Calculate the total cost for each bit_part_id
    df['Estimated cost components (USD)'] = df['Qty'] * df['Cost (USD)']
    total_cost_df = df.groupby(['bit_part_id', 'bit_diameter_in', 'Recon slot'], as_index=False).agg({'Estimated cost components (USD)': 'sum'})
    # Select the relevant columns
    total_cost_df = total_cost_df[['bit_part_id', 'bit_diameter_in', 'Estimated cost components (USD)', 'Recon slot']]
    return total_cost_df


def get_bit_body_cost_to_existing_bits(bit_diameter_in, inv_df):
    bit_diameter_in = float(bit_diameter_in)
    
    # Extract relevant descriptions based on bit diameter size
    if bit_diameter_in >= 12.25:
        descriptions = inv_df['Description'].str.extract(r'(Steel body welded pin diamond HF (\d+(\.\d+)))')
    else:
        descriptions = inv_df['Description'].str.extract(r'(Steel body diamond enhanced HF (\d+(\.\d+)))')
    
    # Drop NaN values and ensure the extracted sizes are floats
    descriptions.dropna(inplace=True)
    descriptions[1] = descriptions[1].astype(float)
    
    # Calculate absolute difference from the target size
    descriptions['difference'] = (descriptions[1] - bit_diameter_in).abs()
    
    # Find the row with the smallest difference
    closest_match = descriptions.sort_values(by='difference').iloc[0]
    # Get the cost of the closest match
    cost = inv_df.loc[inv_df['Description'] == closest_match[0], 'Cost (USD)'].values
    return cost[0] if len(cost) > 0 else 0



def calculate_price_with_markup(df):
    cost = df['Estimated total cost (USD)']
    markup = df['Markup, %']
    df['Price with markup (USD)'] = cost * (1 + markup / 100)
    return df


def calculate_total_price(df):
    external_margin = df['External margin, %']
    df['Total price (USD)'] = df['Price with markup (USD)'] / (1 - (external_margin / 100))
    return df


def calculate_final_base_margin(df):
    df['Final base margin, %'] = ((df['Total price (USD)'] - df['Estimated total cost (USD)']) / df['Estimated total cost (USD)']) * 100
    return df


def apply_discount(df, discount):
    df['Discounted price (USD)'] = df['Total price (USD)'] * (1 - discount / 100)
    return df

