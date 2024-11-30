import base64
import time
import numpy as np
import streamlit as st
st.set_page_config(layout="wide",
                    page_title="Zerdalab Quote Generator",
                    page_icon="images/Zerdalab Logo Colour.png")

from config import Config
from streamlit_cookies_manager import EncryptedCookieManager
import pandas as pd
from exceptions import NotFoundException
from crud.data_lists import (
                            get_cached_bit_sizes,
                            get_cached_dba_data,
                            get_cached_bom_data,
                            choose_bits_from_existing_designs,
                            query_inventory_data,
                            calculate_total_cost,
                            get_bit_body_cost_to_existing_bits,
                            calculate_final_base_margin,
                            calculate_price_with_markup,
                            calculate_total_price,
                            apply_discount
                            )
from whitelist import whitelisted_emails

cookies = EncryptedCookieManager(
    prefix="myapp/", 
    password=Config.COOKIE_SECRET,  
)

if not cookies.ready():
    st.stop()



def main():

    col1, col2 = st.columns([1, 6])   
    with col1: 
        st.markdown(
            """
            <style>
                .center {
                    display: block;
                    margin-left: 10;
                    margin-right: auto;
                    margin-top: -50px;
                    width: 40%;
                    height: auto;
                }
            </style>
            """, unsafe_allow_html=True
            )


        st.markdown(
            """<div class="clickable-image">
                <a href="https://www.zerdalab.com">
                    <img src="data:image/png;base64,{}" class="center">
                </a>
            </div>""".format(
                base64.b64encode(open("images/Zerdalab Logo Colour.png", "rb").read()).decode()
            ),
            unsafe_allow_html=True,
        )


        st.markdown("""
        <style>
            .subheader {
                font-family: Candara, sans-serif;
                font-size: 24px;
                font-weight: 600;
            }
        </style>
        """, unsafe_allow_html=True)


    tab1, tab2, tab3, tab4 = st.tabs([
                                "Inventory price management",
                                "Prices from existing designs",
                                "Prices from mockup designs",
                                "Checkout"])

    # INVENTORY PRICE MANAGEMENT TAB
    with tab1:
        inv_df = query_inventory_data()
        col1, col2 = st.columns(2)

        with col1:
            pid_dropdown = st.multiselect("Select part number", sorted(inv_df['PID'].to_list()))
        with col2:
            desc_dropdown = st.multiselect("Select description", inv_df['Description'].to_list())
        
        # Filter the DataFrame based on the selected values
        if pid_dropdown and desc_dropdown:
            filtered_df = inv_df[(inv_df['PID'].isin(pid_dropdown)) & (inv_df['Description'].isin(desc_dropdown))]
            if filtered_df.empty:
                st.warning("Your search didn't match any results.")
            else:
                st.dataframe(filtered_df)
        elif pid_dropdown:
            filtered_df = inv_df[inv_df['PID'].isin(pid_dropdown)]
            st.dataframe(filtered_df)
        elif desc_dropdown:
            filtered_df = inv_df[inv_df['Description'].isin(desc_dropdown)]
            st.dataframe(filtered_df)
        else:
            st.dataframe(inv_df)


    # PRICES FROM EXISTING DESIGNS TAB
    with tab2:
        bit_sizes = get_cached_bit_sizes()
        
        dba_df = None
        bom_df = None
        existing_bits_df = None

        if 'existing_bits_df' not in st.session_state:
            st.session_state.existing_bits_df = None

        col1, col2 = st.columns(2)

        with col1:
            bit_size_dropdown = st.multiselect("Select bit size", bit_sizes)
            
            if bit_size_dropdown:
                try:
                    dba_df = get_cached_dba_data(bit_size=bit_size_dropdown)   
                    bom_df = get_cached_bom_data(unique_hashes = dba_df['unique_hash'].to_list())
                except NotFoundException as e:
                    st.error(e.details)

        with col2:
            bit_part_id_dropdown = []
            if dba_df is not None:
                bit_part_id_dropdown = st.multiselect("Select bit part id", dba_df['part_number'].to_list())

        if dba_df is not None and bom_df is not None:
            selected_cards = choose_bits_from_existing_designs(dba_df, bom_df, bit_part_id_dropdown)
            if selected_cards:
                new_df = pd.DataFrame(selected_cards)
                add_bits_button = st.button("Add selected bits to cart")
                if add_bits_button:
                    # Ensure existing_bits_df is not None
                    if st.session_state.existing_bits_df is None or st.session_state.existing_bits_df.empty:
                        st.session_state.existing_bits_df = new_df
                    else:
                        # Check if bit_part_id already exists in existing_bits_df
                        existing_bit_part_ids = st.session_state.existing_bits_df['bit_part_id'].unique().tolist()
                        new_bit_part_ids = new_df['bit_part_id'].unique().tolist()
                        duplicate_bit_part_ids = [bit_id for bit_id in new_bit_part_ids if bit_id in existing_bit_part_ids]

                        if duplicate_bit_part_ids:
                            st.warning(f"The following bit_part_ids are already in the cart and will not be added again: {', '.join(duplicate_bit_part_ids)}", icon="⚠️")
                            new_df = new_df[~new_df['bit_part_id'].isin(duplicate_bit_part_ids)]

                        if not new_df.empty:
                            st.session_state.existing_bits_df = pd.concat([st.session_state.existing_bits_df, new_df], ignore_index=True)
                    st.success("Selected bits successfully added to cart. You may proceed to the checkout tab.")
                            

        # Merge existing_bits_df with inv_df to add Cost (USD) column
        if st.session_state.existing_bits_df is not None and not st.session_state.existing_bits_df.empty:
            st.session_state.existing_bits_df = st.session_state.existing_bits_df.drop(columns=['Cost (USD)'], errors='ignore')
            merged_df = st.session_state.existing_bits_df.merge(inv_df[['Description', 'Cost (USD)']], on='Description', how='left')
            st.session_state.existing_bits_df = merged_df
            
            # Calculate the total cost and store it in a separate DataFrame
            st.session_state.total_cost_existing_df = calculate_total_cost(st.session_state.existing_bits_df)

            st.session_state.total_cost_existing_df['bit_body_cost'] = st.session_state.total_cost_existing_df['bit_diameter_in'].apply(lambda x: get_bit_body_cost_to_existing_bits(x, inv_df))

            # Update the total cost with the bit body cost
            # st.session_state.total_cost_existing_df['Estimated total cost (USD)'] += st.session_state.total_cost_existing_df['bit_body_cost']
            
            st.session_state.total_cost_existing_df['Estimated total cost (USD)'] = st.session_state.total_cost_existing_df['Estimated cost components (USD)'] + st.session_state.total_cost_existing_df['bit_body_cost']

            # Update the total cost with logistics, gloves, nozzle wrench, bit breaker and recon slot costs
            gloves_cost = inv_df.loc[inv_df['Description'] == 'Gloves', 'Cost (USD)'].values
            gloves_cost = gloves_cost[0] if len(gloves_cost) > 0 else 0
            nozzle_wrench_cost = inv_df.loc[inv_df['Description'] == 'Nozzle wrench', 'Cost (USD)'].values
            nozzle_wrench_cost = nozzle_wrench_cost[0] if len(nozzle_wrench_cost) > 0 else 0
            logistic_cost = inv_df.loc[inv_df['Description'] == 'Local logistics', 'Cost (USD)'].values
            local_logistics_cost = logistic_cost[0] if len(logistic_cost) > 0 else 0
            recon = inv_df.loc[inv_df['Description'] == 'Recon slot', 'Cost (USD)'].values
            recon_cost = recon[0] if len(recon) > 0 else 0
            # Get the costs for the bit boxes
            bit_box_steel_cost = inv_df.loc[inv_df['Description'] == 'Bit box steel', 'Cost (USD)'].values
            bit_box_steel_cost = bit_box_steel_cost[0] if len(bit_box_steel_cost) > 0 else 0
            bit_box_plastic_cost = inv_df.loc[inv_df['Description'] == 'Bit box plastic', 'Cost (USD)'].values
            bit_box_plastic_cost = bit_box_plastic_cost[0] if len(bit_box_plastic_cost) > 0 else 0
            bit_box_wood_cost = inv_df.loc[inv_df['Description'] == 'Bit box wood', 'Cost (USD)'].values
            bit_box_wood_cost = bit_box_wood_cost[0] if len(bit_box_wood_cost) > 0 else 0

            # Add the bit breaker cost
            bit_breaker_large_cost = inv_df.loc[inv_df['Description'] == 'Bit breaker large', 'Cost (USD)'].values
            bit_breaker_large_cost = bit_breaker_large_cost[0] if len(bit_breaker_large_cost) > 0 else 0
            bit_breaker_small_cost = inv_df.loc[inv_df['Description'] == 'Bit breaker small', 'Cost (USD)'].values
            bit_breaker_small_cost = bit_breaker_small_cost[0] if len(bit_breaker_small_cost) > 0 else 0

            # Define the conditions and corresponding choices for the bit box costs
            conditions = [
                st.session_state.total_cost_existing_df['bit_diameter_in'] >= 12.25,
                st.session_state.total_cost_existing_df['bit_diameter_in'] < 12.25
            ]
            choices = [bit_box_steel_cost, 
                        bit_box_plastic_cost]

            # Apply the conditions to get the bit box cost
            st.session_state.total_cost_existing_df['bit_box_cost'] = np.select(conditions, choices, default=0)
            # Apply the condition to get the bit breaker cost
            st.session_state.total_cost_existing_df['bit_breaker_cost'] = np.where(st.session_state.total_cost_existing_df['bit_diameter_in'] >= 16, 
                                                                                bit_breaker_large_cost,
                                                                                bit_breaker_small_cost)
            
            st.session_state.total_cost_existing_df['Estimated total cost (USD)'] += (
                gloves_cost +
                local_logistics_cost +
                nozzle_wrench_cost +
                np.where(st.session_state.total_cost_existing_df['Recon slot'].str.startswith('Recon', na=False), recon_cost, 0) +
                st.session_state.total_cost_existing_df['bit_box_cost'] +
                st.session_state.total_cost_existing_df['bit_breaker_cost']
            )
            # Add a horizontal line
            # st.write(st.session_state.total_cost_existing_df)
            st.markdown("---")
            st.dataframe(st.session_state.total_cost_existing_df[['bit_part_id', 'bit_diameter_in','Estimated total cost (USD)']], hide_index=True)


    # PRICES FROM MOCKUP DESIGNS TAB   
    with tab3:

        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 0.1, 1, 0.3, 0.1, 1, 0.3])
        with col1:
            mockup_desc = st.text_input("Description for mockup bit")
            bit_dia = st.number_input("Bit diameter", min_value=0.000, max_value=100.000, value=0.000, step=0.001, format="%.3f")
            body_descriptions = [desc for desc in inv_df['Description'].to_list() if 'body' in desc.lower()]
            bit_body = st.selectbox("Bit body", body_descriptions)

        with col3:
            nozzle_descriptions = [desc for desc in inv_df['Description'].to_list() if ('nozzle' in desc.lower() or 'port' in desc.lower()) and 'nozzle wrench' not in desc.lower()]
            nozzle_type = st.selectbox("Nozzle/Port", nozzle_descriptions)
            cutter_description = [desc for desc in inv_df['Description'].to_list() if 'cutter' in desc.lower()]
            cutter_type = st.selectbox("Cutter", cutter_description)
            recon_needed = st.selectbox("Recon slot needed", ["No", "Yes"])
        
        with col4:
            number_of_nozzles = st.number_input("Qty nozzles", min_value=0, max_value=1000, value=0, step=1)
            number_of_cutters = st.number_input("Qty cutters", min_value=0, max_value=1000, value=0, step=1)
        
        with col6:
            sec_component_descriptions = [desc for desc in inv_df['Description'].to_list()
                                        if 'conical' in desc.lower() or
                                            'pyramid' in desc.lower() or 
                                            'dome' in desc.lower()]
            sec_component_1 = st.selectbox("Secondary Component 1", sec_component_descriptions)
            sec_component_2 = st.selectbox("Secondary Component 2", sec_component_descriptions)
            sec_component_3 = st.selectbox("Secondary Component 3", sec_component_descriptions)
            
        with col7:
            number_of_sec_comp_1 = st.number_input("Qty SC 1", min_value=0, max_value=1000, value=0, step=1)
            number_of_sec_comp_2 = st.number_input("Qty SC 2", min_value=0, max_value=1000, value=0, step=1)
            number_of_sec_comp_3 = st.number_input("Qty SC 3", min_value=0, max_value=1000, value=0, step=1)

        add_button = st.button("Add mockup data")

        if add_button:
            mockup_data = {
                "bit_part_id": mockup_desc,
                "bit_diameter_in": bit_dia,
                "bit_body": bit_body,
                "cutter_type": cutter_type,
                "number_of_cutters": number_of_cutters,
                "nozzle_type": nozzle_type,
                "number_of_nozzles": number_of_nozzles,
                "sec_component_1": sec_component_1,
                "number_of_sec_comp_1": number_of_sec_comp_1,
                "sec_component_2": sec_component_2,
                "number_of_sec_comp_2": number_of_sec_comp_2,
                "sec_component_3": sec_component_3,
                "number_of_sec_comp_3": number_of_sec_comp_3,
                "Recon slot needed": recon_needed
            }
            new_df = pd.DataFrame([mockup_data])

            # Merge mockup_data with inv_df to get the costs
            merged_df = new_df.merge(inv_df, left_on='cutter_type', right_on='Description', how='left', suffixes=('', '_cutter'))
            merged_df = merged_df.merge(inv_df, left_on='nozzle_type', right_on='Description', how='left', suffixes=('', '_nozzle'))
            merged_df = merged_df.merge(inv_df, left_on='sec_component_1', right_on='Description', how='left', suffixes=('', '_sec1'))
            merged_df = merged_df.merge(inv_df, left_on='sec_component_2', right_on='Description', how='left', suffixes=('', '_sec2'))
            merged_df = merged_df.merge(inv_df, left_on='sec_component_3', right_on='Description', how='left', suffixes=('', '_sec3'))
            merged_df = merged_df.merge(inv_df, left_on='bit_body', right_on='Description', how='left', suffixes=('', '_body'))

            # Drop the extra columns
            merged_df = merged_df.drop(columns=['PID','PID_sec1', 'PID_sec2', 'PID_sec3', 'PID_nozzle'], errors='ignore')
            
            gloves_cost = inv_df.loc[inv_df['Description'] == 'Gloves', 'Cost (USD)'].values
            gloves_cost = gloves_cost[0] if len(gloves_cost) > 0 else 0
            nozzle_wrench_cost = inv_df.loc[inv_df['Description'] == 'Nozzle wrench', 'Cost (USD)'].values
            nozzle_wrench_cost = nozzle_wrench_cost[0] if len(nozzle_wrench_cost) > 0 else 0
            logistic_cost = inv_df.loc[inv_df['Description'] == 'Local logistics', 'Cost (USD)'].values
            local_logistics_cost = logistic_cost[0] if len(logistic_cost) > 0 else 0
            recon = inv_df.loc[inv_df['Description'] == 'Recon slot', 'Cost (USD)'].values
            recon_cost = recon[0] if len(recon) > 0 else 0

            # Get the costs for the bit boxes
            bit_box_steel_cost = inv_df.loc[inv_df['Description'] == 'Bit box steel', 'Cost (USD)'].values
            bit_box_steel_cost = bit_box_steel_cost[0] if len(bit_box_steel_cost) > 0 else 0
            bit_box_plastic_cost = inv_df.loc[inv_df['Description'] == 'Bit box plastic', 'Cost (USD)'].values
            bit_box_plastic_cost = bit_box_plastic_cost[0] if len(bit_box_plastic_cost) > 0 else 0
            bit_box_wood_cost = inv_df.loc[inv_df['Description'] == 'Bit box wood', 'Cost (USD)'].values
            bit_box_wood_cost = bit_box_wood_cost[0] if len(bit_box_wood_cost) > 0 else 0

            # Add the bit breaker cost
            bit_breaker_large_cost = inv_df.loc[inv_df['Description'] == 'Bit breaker large', 'Cost (USD)'].values
            bit_breaker_large_cost = bit_breaker_large_cost[0] if len(bit_breaker_large_cost) > 0 else 0
            bit_breaker_small_cost = inv_df.loc[inv_df['Description'] == 'Bit breaker small', 'Cost (USD)'].values
            bit_breaker_small_cost = bit_breaker_small_cost[0] if len(bit_breaker_small_cost) > 0 else 0

            # Define the conditions and corresponding choices for the bit box costs
            conditions = [
                merged_df['bit_diameter_in'] >= 12.25,
                merged_df['bit_diameter_in'] < 12.25
            ]
            choices = [bit_box_steel_cost, 
                        bit_box_plastic_cost]

            # Apply the conditions to get the bit box cost
            merged_df['bit_box_cost'] = np.select(conditions, choices, default=0)
            # Apply the condition to get the bit breaker cost
            merged_df['bit_breaker_cost'] = np.where(merged_df['bit_diameter_in'] >= 16, bit_breaker_large_cost, bit_breaker_small_cost)

            # Calculate the total cost
            merged_df['Estimated total cost (USD)'] = (
                merged_df['Cost (USD)'] * merged_df['number_of_cutters'] +
                merged_df['Cost (USD)_nozzle'] * merged_df['number_of_nozzles'] +
                merged_df['Cost (USD)_sec1'] * merged_df['number_of_sec_comp_1'] +
                merged_df['Cost (USD)_sec2'] * merged_df['number_of_sec_comp_2'] +
                merged_df['Cost (USD)_sec3'] * merged_df['number_of_sec_comp_3'] +
                merged_df['Cost (USD)_body'] + 
                gloves_cost +
                local_logistics_cost +
                nozzle_wrench_cost +
                np.where(merged_df['Recon slot needed'] == 'Yes', recon_cost, 0) +
                merged_df['bit_box_cost'] +
                merged_df['bit_breaker_cost']
            )

            reduced_df = merged_df[['bit_part_id', 'bit_diameter_in', 'Estimated total cost (USD)']]
        
            # Check if the session state already contains a DataFrame
            if 'mockup_data_df' in st.session_state:
                # Append the new data to the existing DataFrame
                st.session_state.mockup_data_df = pd.concat([st.session_state.mockup_data_df, reduced_df], ignore_index=True)
            else:
                # Create a new DataFrame in the session state
                st.session_state.mockup_data_df = reduced_df
            
            success_message_mockup = st.success("Selected bits successfully added to cart. You may proceed to the checkout tab.")
            # time.sleep(2)
            # success_message_mockup.empty()

        # Display the DataFrame stored in the session state
        if 'mockup_data_df' in st.session_state and not st.session_state.mockup_data_df.empty:
            # Add a horizontal line
            st.markdown("---")
            st.dataframe(st.session_state.mockup_data_df, hide_index=True)

            # Add Delete and Reset buttons under the table
            col1, col2, col3 = st.columns([1,1,4])
            with col1:
                if st.button("Delete Last Added Data"):
                    if not st.session_state.mockup_data_df.empty:
                        st.session_state.mockup_data_df = st.session_state.mockup_data_df.iloc[:-1]
                        st.rerun()  
                    else:
                        st.warning("No data to delete.")

            with col2:
                if st.button("Reset Data"):
                    st.session_state.mockup_data_df = pd.DataFrame()
                    st.rerun()


    # CHECKOUT TAB
    with tab4:
        # Ensure the total_cost_existing_df is correctly initialized
        if (
            "existing_bits_df" in st.session_state
            and st.session_state.existing_bits_df is not None
            and not st.session_state.existing_bits_df.empty
        ):
            st.session_state.total_cost_existing_df = st.session_state.total_cost_existing_df[
                ["bit_part_id", "bit_diameter_in", "Estimated total cost (USD)"]
            ]

        # Initialize mockup_data_df if not already present
        if "mockup_data_df" not in st.session_state:
            st.session_state.mockup_data_df = pd.DataFrame()

        # Prepare the final_df by merging or selecting available DataFrames
        final_dfs = []
        if (
            "existing_bits_df" in st.session_state
            and st.session_state.existing_bits_df is not None
            and not st.session_state.existing_bits_df.empty
        ):
            final_dfs.append(st.session_state.total_cost_existing_df)

        if (
            "mockup_data_df" in st.session_state
            and st.session_state.mockup_data_df is not None
            and not st.session_state.mockup_data_df.empty
        ):
            final_dfs.append(st.session_state.mockup_data_df)

        if final_dfs:

            # Define a callback function to recalculate the prices
            def recalculate_prices():
                print("Recalculating prices...")

                st.session_state.final_df = calculate_price_with_markup(st.session_state.final_df)
                st.session_state.final_df = calculate_total_price(st.session_state.final_df)
                st.session_state.final_df = calculate_final_base_margin(st.session_state.final_df)

                # Rearrange the column sequence
                desired_order = [
                    "part_id",
                    "diameter_in",
                    "Total Cost (USD)",
                    "Markup, %",
                    "Price with markup (USD)",
                    "External margin, %",
                    "Total price (USD)",
                    "Final base margin, %",
                    "Select"
                ]
                st.session_state.final_df = st.session_state.final_df[desired_order]


            # Initialize the final DataFrame if not already present
            if 'final_df' not in st.session_state:
                st.session_state.final_df = pd.concat(final_dfs, ignore_index=True)

                # Add default values and extra columns
                if "Markup, %" not in st.session_state.final_df.columns:
                    st.session_state.final_df["Markup, %"] = 30
                if "External margin, %" not in st.session_state.final_df.columns:
                    st.session_state.final_df["External margin, %"] = 50
                if "Select" not in st.session_state.final_df.columns:
                    st.session_state.final_df["Select"] = False

                recalculate_prices()

            # Define the column configuration
            column_config = {
                "part_id": st.column_config.TextColumn(
                    "Part ID",
                    disabled=True,
                ),
                "diameter_in": st.column_config.NumberColumn(
                    "Diameter",
                    disabled=True,
                ),
                "Total Cost (USD)": st.column_config.NumberColumn(
                    "Total Cost (USD)",
                    disabled=True,
                ),
                "Markup, %": st.column_config.NumberColumn(
                    "Markup, %",
                ),
                "External margin, %": st.column_config.NumberColumn(
                    "External margin, %",

                ),
                "Price with markup (USD)": st.column_config.NumberColumn(
                    "Price with markup (USD)",
                    disabled=True,

                ),
                "Total price (USD)": st.column_config.NumberColumn(
                    "Total price (USD)",
                    disabled=True,
                ),
                "Final base margin, %": st.column_config.NumberColumn(
                    "Final base margin, %",
                    disabled=True,
                ),

                "Select": st.column_config.CheckboxColumn("Select"),
            }

            # Display the data editor
            edited_df = st.data_editor(
                st.session_state.final_df,
                key="data_editor",
                hide_index=True,
                column_config=column_config,
            )

            # Check if the data has been edited and recalculate prices
            if not edited_df.equals(st.session_state.final_df):
                st.session_state.final_df = edited_df
                recalculate_prices()

                # Trigger a rerun to update the displayed data
                st.rerun()

            # Add a delete button to remove selected rows
            if st.button("Delete selected bits"):
                edited_rows = st.session_state["data_editor"].get("edited_rows", {})
                rows_to_delete = [idx for idx, value in edited_rows.items() if value.get("Select")]

                # If rows to delete exist
                if rows_to_delete:
                    # Identify bit_part_ids to remove
                    bit_part_ids_to_remove = st.session_state.final_df.loc[
                        rows_to_delete, "bit_part_id"
                    ].tolist()

                    # Update the final_df in the session state
                    st.session_state.final_df = st.session_state.final_df[
                        ~st.session_state.final_df["bit_part_id"].isin(bit_part_ids_to_remove)
                    ].reset_index(drop=True)

                    # Update existing_bits_df and mockup_data_df accordingly
                    if (
                        "existing_bits_df" in st.session_state
                        and st.session_state.existing_bits_df is not None
                        and "bit_part_id" in st.session_state.existing_bits_df.columns
                    ):
                        st.session_state.existing_bits_df = st.session_state.existing_bits_df[
                            ~st.session_state.existing_bits_df["bit_part_id"].isin(bit_part_ids_to_remove)
                        ].reset_index(drop=True)

                    if (
                        "mockup_data_df" in st.session_state
                        and st.session_state.mockup_data_df is not None
                        and "bit_part_id" in st.session_state.mockup_data_df.columns
                    ):
                        st.session_state.mockup_data_df = st.session_state.mockup_data_df[
                            ~st.session_state.mockup_data_df["bit_part_id"].isin(bit_part_ids_to_remove)
                        ].reset_index(drop=True)

                    # Rerun the script to reflect changes
                    st.rerun()

        else:
            st.warning("No bits have been added to the cart.")



def login_form():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form(key='login_form'):
            user_email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            if login_button:
                if user_email == "" or password == "":
                    st.write("Please enter an email and password")
                else:
                    if user_email in whitelisted_emails and password == Config.LOGIN_PASSWORD:
                        st.session_state.logged_in = True
                        cookies["logged_in"] = "true"
                        cookies.save()
                        st.rerun()
                    else:
                        st.write("Invalid email or password")



def app():
    # Ensure that the internal table exists, this should not be used on bit_data or habitat tables,
    # st.set_page_config(layout="wide")
    st.markdown("<h4 style='text-align: center; color: black;'>Zerdalab Quote Estimator</h4>",
                unsafe_allow_html=True)

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Check the cookie to see if the user is logged in
    if cookies.get("logged_in") == "true":
        st.session_state.logged_in = True

    if not st.session_state.logged_in:
        login_form()

    else:
        col1, col2 = st.columns([8, 1])
        st.write("")
        with col2:
            logout = st.button("Logout")
        if logout:
            st.session_state.logged_in = False
            cookies["logged_in"] = "false"
            cookies.save()
            st.rerun()

        main()


if __name__ == "__main__":
    app()