# Use-case coverage — where new data would diversify the training set

Generated 2026-08-13 by `scripts/usecase_coverage.py` on `jasper`, under the **cleantech** anchor, against **34 existing use cases** and **79 candidate themes** from `scripts/usecase_themes.json`.

> **Candidate themes have briefs, not corpora.** Every score here is brief-space geometry standing in for where a use case's papers would land. Measured across the 34 use cases that have both, that substitution holds at Spearman 0.60 — real, but loose. Read this as a shortlist to go and acquire, not as a measurement. A `duplicate-risk` flag is a prompt to check against the real corpus, not a verdict.

## Calibration (computed at run time, not hardcoded)

| Landmark | Value | What it means |
|---|---|---|
| Domain pass mark | 0.342 | Weakest `in_domain_reference` use case (`cement_binders`). |
| Domain floor (guard rail) | 0.173 | Midpoint between your industrial and biomedical use cases. Below it is `off-domain`. This axis is a guard rail, not a ranking — see the stability note below. |
| Existing nearest-neighbour, median | 0.248 | How close a typical pair of use cases you already have sits. |
| Existing nearest-neighbour, max | 0.711 | The closest real pair (`synergy_hall_2012 / synergy_radjenovic_2013`) — the only pair ever measured to inflate a LOGO score. Nothing here should be allowed to reach it. |
| `overlaps-existing` line | p75 | Closer to something you own than 75% of your use cases are to their own nearest neighbour. |
| `duplicate-risk` line | p90 | Same at 90%. Check against the real corpus before acquiring. |

**Verdicts:** adopt 57, duplicate-risk 1, off-domain 6, overlaps-existing 15

### Where your existing use cases sit against the domain anchor

The guard rail is the **domain floor**, not the pass mark: the two `in_domain_reference` use cases only serve to locate the industrial mode, and everything in that mode passes. Use cases sitting between the floor and the pass mark are in-domain but less central than your reference pair — a fact about the current portfolio, not a problem.

| Use case | domain_fit | vs guard rail (0.173) |
|---|---|---|
| `tech_forecasting` | 0.415 | in domain |
| `carbon_capture` | 0.414 | in domain |
| `cement_binders` | 0.342 | in domain |
| `roadfreight_metareview` | 0.301 | in domain |
| `solar_leo` | 0.238 | in domain |
| `soil_microbiome` | 0.149 | off-domain |
| `nykvist_evcharging` | 0.117 | off-domain |
| `ner` | 0.101 | off-domain |
| `synergy_menon_2022` | 0.059 | off-domain |
| `synergy_walker_2018` | -0.026 | off-domain |
| *(remaining 24 — all benchset biomedical/psychology collections)* | -0.161 to -0.027 | below |

## Which sectors are open ground — **this is the output to act on**

`median_overlap_pct` is how close the sector's themes typically sit to the nearest thing you already have — **lower means more new ground**. This ranking is the most stable thing the tool produces: Jasper and Qwen3-4B agree on it at Spearman **0.97**, against 91% agreement on individual verdicts and only 4/12 on the ordered portfolio below. Decide the *sector* here; pick themes within it for reasons this tool cannot see (data availability, customer demand, labelling cost).

| sector                     |   candidates |   open_ground |   median_overlap_pct |   median_domain_fit |
|:---------------------------|-------------:|--------------:|---------------------:|--------------------:|
| Manufacturing & automation |            7 |             5 |                0.206 |               0.221 |
| Circularity & materials    |            3 |             3 |                0.235 |               0.348 |
| Materials & durability     |            5 |             3 |                0.235 |               0.218 |
| Process industries         |            4 |             4 |                0.235 |               0.278 |
| Water, waste & emissions   |            6 |             6 |                0.265 |               0.382 |
| Oil, gas & subsea          |            5 |             3 |                0.294 |               0.215 |
| Measurement & accounting   |            3 |             2 |                0.324 |               0.429 |
| Metals & mining            |            8 |             8 |                0.574 |               0.272 |
| Energy & power             |            8 |             6 |                0.588 |               0.331 |
| Non-CO2 emissions          |            2 |             2 |                0.618 |               0.34  |
| Clean energy carriers      |            4 |             2 |                0.647 |               0.334 |
| Chemicals & process        |            6 |             4 |                0.647 |               0.355 |
| Industrial energy systems  |            4 |             4 |                0.676 |               0.38  |
| Transport & logistics      |            6 |             3 |                0.735 |               0.262 |
| Construction materials     |            3 |             1 |                0.765 |               0.28  |
| Carbon removal & capture   |            5 |             1 |                0.882 |               0.358 |

## An illustrative portfolio (top 12) — **not a procurement plan**

Greedy farthest-point selection: each pick is the candidate furthest from everything chosen so far, so the set spreads across *different* gaps instead of crowding into the largest one. `min_cos_to_set` is the pick's similarity to its closest neighbour at the moment it was chosen — lower means more new ground.

**Read this as one valid way to cover the open sectors, not as a ranking.** Greedy selection is chaotic: each pick reshuffles the remaining scores, so the two embedding models produce lists sharing only 4 of 12 entries while still agreeing on which sectors those entries come from.

|   pick | theme                            | sector                     | use_case_name                                                |   min_cos_to_set | nearest_existing     |   nearest_cos |   domain_fit |
|-------:|:---------------------------------|:---------------------------|:-------------------------------------------------------------|-----------------:|:---------------------|--------------:|-------------:|
|      1 | pharma_continuous_manufacturing  | Process industries         | Continuous Pharmaceutical Manufacturing                      |            0.122 | synergy_donners_2021 |         0.122 |        0.18  |
|      2 | ewaste_metal_recovery            | Water, waste & emissions   | Critical Metal Recovery from Electronic Waste                |            0.148 | carbon_capture       |         0.148 |        0.199 |
|      3 | industrial_robotics              | Manufacturing & automation | Industrial Robot Manipulation in Unstructured Environments   |            0.182 | tech_forecasting     |         0.182 |        0.317 |
|      4 | lca_industrial                   | Measurement & accounting   | Life Cycle Assessment of Industrial Products                 |            0.21  | synergy_menon_2022   |         0.203 |        0.291 |
|      5 | corrosion_coatings               | Materials & durability     | Protective Coatings Against Industrial Corrosion             |            0.218 | carbon_capture       |         0.156 |        0.218 |
|      6 | grid_storage                     | Energy & power             | Grid-Scale Long-Duration Energy Storage                      |            0.229 | solar_leo            |         0.226 |        0.335 |
|      7 | methane_leak_detection           | Oil, gas & subsea          | Methane Leak Detection and Quantification                    |            0.242 | carbon_capture       |         0.152 |        0.215 |
|      8 | refinery_catalysts               | Oil, gas & subsea          | Refinery Hydroprocessing Catalyst Performance                |            0.25  | carbon_capture       |         0.229 |        0.292 |
|      9 | geothermal_drilling              | Energy & power             | Deep Geothermal Drilling Technology                          |            0.267 | carbon_capture       |         0.255 |        0.254 |
|     10 | predictive_maintenance_vibration | Manufacturing & automation | Vibration-Based Predictive Maintenance of Rotating Equipment |            0.279 | synergy_hall_2012    |         0.274 |        0.221 |
|     11 | biochar_carbon_removal           | Carbon removal & capture   | Biochar Production and Carbon Permanence                     |            0.298 | soil_microbiome      |         0.298 |        0.305 |
|     12 | aluminium_inert_anode            | Metals & mining            | Inert Anode Aluminium Smelting                               |            0.303 | carbon_capture       |         0.303 |        0.345 |

## Every candidate, by sector

### Carbon removal & capture

| theme                  | use_case_name                                    | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:-----------------------|:-------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| biochar_carbon_removal | Biochar Production and Carbon Permanence         | adopt             | soil_microbiome    |         0.298 |        0.305 |
| enhanced_weathering    | Enhanced Rock Weathering and Mineral Carbonation | overlaps-existing | carbon_capture     |         0.351 |        0.336 |
| cement_kiln_ccs        | Cement Kiln Carbon Capture                       | overlaps-existing | carbon_capture     |         0.449 |        0.434 |
| co2_concrete_curing    | CO2 Mineralisation and Curing in Concrete        | overlaps-existing | cement_binders     |         0.471 |        0.358 |
| direct_air_capture     | Direct Air Capture of CO2                        | duplicate-risk    | carbon_capture     |         0.552 |        0.476 |

### Chemicals & process

| theme                   | use_case_name                                      | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:------------------------|:---------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| solvent_recovery        | Industrial Solvent Recovery and Reuse              | adopt             | carbon_capture     |         0.218 |        0.422 |
| chloralkali_efficiency  | Chlor-Alkali Electrolysis Efficiency               | adopt             | carbon_capture     |         0.25  |        0.339 |
| process_intensification | Continuous Flow Process Intensification            | adopt             | carbon_capture     |         0.257 |        0.371 |
| membrane_separation     | Industrial Membrane Gas Separation                 | adopt             | carbon_capture     |         0.313 |        0.41  |
| green_ammonia           | Low-Pressure and Electrochemical Ammonia Synthesis | overlaps-existing | carbon_capture     |         0.382 |        0.292 |
| methanol_co2            | CO2-to-Methanol Catalytic Synthesis                | overlaps-existing | carbon_capture     |         0.39  |        0.302 |

### Circularity & materials

| theme                         | use_case_name                                     | verdict   | nearest_existing   |   nearest_cos |   domain_fit |
|:------------------------------|:--------------------------------------------------|:----------|:-------------------|--------------:|-------------:|
| circular_product_design       | Design for Disassembly and Remanufacturing        | adopt     | carbon_capture     |         0.18  |        0.348 |
| battery_recycling             | Lithium-Ion Battery Recycling                     | adopt     | carbon_capture     |         0.198 |        0.248 |
| critical_mineral_substitution | Critical Mineral Substitution in Clean Technology | adopt     | carbon_capture     |         0.301 |        0.393 |

### Clean energy carriers

| theme                       | use_case_name                                       | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:----------------------------|:----------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| green_hydrogen_electrolysis | Water Electrolysis Stack Performance and Durability | adopt             | tech_forecasting   |         0.243 |        0.337 |
| biomethane_upgrading        | Anaerobic Digestion and Biomethane Upgrading        | adopt             | carbon_capture     |         0.253 |        0.284 |
| methane_pyrolysis           | Methane Pyrolysis for Hydrogen and Solid Carbon     | overlaps-existing | carbon_capture     |         0.33  |        0.332 |
| efuels_synthesis            | E-Fuel Synthesis from Captured CO2                  | overlaps-existing | carbon_capture     |         0.376 |        0.353 |

### Construction materials

| theme              | use_case_name                                      | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:-------------------|:---------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| refractory_linings | High-Temperature Refractory Lining Durability      | adopt             | cement_binders     |         0.32  |        0.29  |
| concrete_recycling | Structural Reuse of Recycled Concrete Aggregate    | overlaps-existing | cement_binders     |         0.332 |        0.253 |
| calcined_clay_scm  | Calcined Clay Supplementary Cementitious Materials | overlaps-existing | cement_binders     |         0.466 |        0.28  |

### Energy & power

| theme                           | use_case_name                                       | verdict           | nearest_existing       |   nearest_cos |   domain_fit |
|:--------------------------------|:----------------------------------------------------|:------------------|:-----------------------|--------------:|-------------:|
| offshore_wind_foundations       | Offshore Wind Foundation Engineering                | off-domain        | carbon_capture         |         0.182 |        0.169 |
| grid_storage                    | Grid-Scale Long-Duration Energy Storage             | adopt             | solar_leo              |         0.226 |        0.335 |
| smr_nuclear                     | Small Modular Reactor Deployment                    | adopt             | tech_forecasting       |         0.232 |        0.303 |
| geothermal_drilling             | Deep Geothermal Drilling Technology                 | adopt             | carbon_capture         |         0.255 |        0.254 |
| hydrogen_storage_transport      | Hydrogen Storage and Pipeline Transport             | adopt             | roadfreight_metareview |         0.269 |        0.327 |
| waste_heat_recovery             | Industrial Waste Heat Recovery                      | adopt             | carbon_capture         |         0.27  |        0.405 |
| high_temp_heat_pumps            | Industrial High-Temperature Heat Pumps              | adopt             | carbon_capture         |         0.275 |        0.359 |
| industrial_heat_electrification | Electrification of High-Temperature Industrial Heat | overlaps-existing | tech_forecasting       |         0.357 |        0.477 |

### Industrial energy systems

| theme                      | use_case_name                                    | verdict   | nearest_existing   |   nearest_cos |   domain_fit |
|:---------------------------|:-------------------------------------------------|:----------|:-------------------|--------------:|-------------:|
| thermal_energy_storage     | Industrial Thermal Energy Storage                | adopt     | carbon_capture     |         0.273 |        0.404 |
| solar_thermal_industrial   | Concentrated Solar Heat for Industrial Processes | adopt     | solar_leo          |         0.29  |        0.356 |
| industrial_symbiosis       | Industrial Symbiosis and Waste Heat Networks     | adopt     | soil_microbiome    |         0.295 |        0.472 |
| industrial_demand_response | Industrial Demand-Side Flexibility               | adopt     | nykvist_evcharging |         0.303 |        0.31  |

### Manufacturing & automation

| theme                            | use_case_name                                                | verdict    | nearest_existing   |   nearest_cos |   domain_fit |
|:---------------------------------|:-------------------------------------------------------------|:-----------|:-------------------|--------------:|-------------:|
| welding_automation               | Automated and Robotic Welding Process Control                | off-domain | cement_binders     |         0.133 |        0.157 |
| additive_manufacturing_metal     | Metal Additive Manufacturing Qualification                   | adopt      | tech_forecasting   |         0.181 |        0.221 |
| industrial_robotics              | Industrial Robot Manipulation in Unstructured Environments   | adopt      | tech_forecasting   |         0.182 |        0.317 |
| digital_twin_process             | Digital Twins for Process Plant Operations                   | adopt      | tech_forecasting   |         0.185 |        0.297 |
| ndt_inspection                   | Non-Destructive Testing of Welds and Castings                | off-domain | synergy_hall_2012  |         0.202 |        0.098 |
| machine_vision_qc                | Machine Vision Surface Defect Inspection                     | adopt      | synergy_hall_2012  |         0.249 |        0.196 |
| predictive_maintenance_vibration | Vibration-Based Predictive Maintenance of Rotating Equipment | adopt      | synergy_hall_2012  |         0.274 |        0.221 |

### Materials & durability

| theme                 | use_case_name                                    | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:----------------------|:-------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| corrosion_coatings    | Protective Coatings Against Industrial Corrosion | adopt             | carbon_capture     |         0.156 |        0.218 |
| weld_fatigue          | Fatigue Life of Welded Steel Structures          | off-domain        | cement_binders     |         0.171 |        0.071 |
| tribology_lubricants  | Industrial Lubricants and Tribological Wear      | adopt             | cement_binders     |         0.211 |        0.299 |
| high_temp_alloys      | High-Temperature Structural Alloys               | adopt             | cement_binders     |         0.253 |        0.187 |
| structural_composites | Fibre-Reinforced Composites for Heavy Structures | overlaps-existing | cement_binders     |         0.322 |        0.254 |

### Measurement & accounting

| theme                   | use_case_name                                               | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:------------------------|:------------------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| lca_industrial          | Life Cycle Assessment of Industrial Products                | adopt             | synergy_menon_2022 |         0.203 |        0.291 |
| carbon_mrv              | Industrial Emissions Monitoring, Reporting and Verification | adopt             | tech_forecasting   |         0.234 |        0.429 |
| industrial_water_energy | Water and Energy Nexus in Industrial Decarbonisation        | overlaps-existing | tech_forecasting   |         0.358 |        0.593 |

### Metals & mining

| theme                 | use_case_name                                           | verdict   | nearest_existing       |   nearest_cos |   domain_fit |
|:----------------------|:--------------------------------------------------------|:----------|:-----------------------|--------------:|-------------:|
| mine_automation       | Autonomous Underground Mining Equipment                 | adopt     | roadfreight_metareview |         0.183 |        0.2   |
| eaf_scrap_quality     | Electric Arc Furnace Scrap Contamination Control        | adopt     | cement_binders         |         0.228 |        0.267 |
| rare_earth_separation | Rare Earth Element Separation and Recovery              | adopt     | carbon_capture         |         0.239 |        0.238 |
| copper_hydromet       | Hydrometallurgical Copper Extraction from Low-Grade Ore | adopt     | carbon_capture         |         0.254 |        0.226 |
| tailings_reprocessing | Mine Tailings Reprocessing and Stabilisation            | adopt     | carbon_capture         |         0.26  |        0.277 |
| ore_sorting_sensing   | Sensor-Based Ore Sorting                                | adopt     | carbon_capture         |         0.261 |        0.28  |
| steel_h2_dri          | Hydrogen Direct Reduction Ironmaking                    | adopt     | carbon_capture         |         0.3   |        0.377 |
| aluminium_inert_anode | Inert Anode Aluminium Smelting                          | adopt     | carbon_capture         |         0.303 |        0.345 |

### Non-CO2 emissions

| theme          | use_case_name                                  | verdict   | nearest_existing   |   nearest_cos |   domain_fit |
|:---------------|:-----------------------------------------------|:----------|:-------------------|--------------:|-------------:|
| n2o_abatement  | Nitrous Oxide Abatement in Chemical Production | adopt     | carbon_capture     |         0.271 |        0.351 |
| fgas_abatement | SF6 and Fluorinated Gas Alternatives           | adopt     | cement_binders     |         0.275 |        0.328 |

### Oil, gas & subsea

| theme                  | use_case_name                                 | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:-----------------------|:----------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| pipeline_integrity     | Pipeline Integrity and Corrosion Monitoring   | off-domain        | nykvist_evcharging |         0.149 |        0.146 |
| methane_leak_detection | Methane Leak Detection and Quantification     | adopt             | carbon_capture     |         0.152 |        0.215 |
| subsea_inspection      | Subsea Asset Inspection Robotics              | adopt             | solar_leo          |         0.217 |        0.184 |
| refinery_catalysts     | Refinery Hydroprocessing Catalyst Performance | adopt             | carbon_capture     |         0.229 |        0.292 |
| flare_reduction        | Flare Gas Reduction and Utilisation           | overlaps-existing | carbon_capture     |         0.344 |        0.355 |

### Process industries

| theme                           | use_case_name                                     | verdict   | nearest_existing       |   nearest_cos |   domain_fit |
|:--------------------------------|:--------------------------------------------------|:----------|:-----------------------|--------------:|-------------:|
| pharma_continuous_manufacturing | Continuous Pharmaceutical Manufacturing           | adopt     | synergy_donners_2021   |         0.122 |        0.18  |
| textile_recycling               | Fibre-to-Fibre Textile Recycling                  | adopt     | soil_microbiome        |         0.192 |        0.295 |
| food_processing_energy          | Energy Efficiency in Food and Beverage Processing | adopt     | roadfreight_metareview |         0.203 |        0.378 |
| lignin_valorisation             | Lignin Valorisation from Pulp and Paper           | adopt     | cement_binders         |         0.214 |        0.261 |

### Transport & logistics

| theme                       | use_case_name                                       | verdict           | nearest_existing       |   nearest_cos |   domain_fit |
|:----------------------------|:----------------------------------------------------|:------------------|:-----------------------|--------------:|-------------:|
| saf_aviation                | Sustainable Aviation Fuel Production                | adopt             | roadfreight_metareview |         0.251 |        0.265 |
| cold_chain_refrigeration    | Industrial Cold Chain Refrigeration                 | adopt             | roadfreight_metareview |         0.255 |        0.309 |
| maritime_alt_fuels          | Ammonia and Methanol Marine Engines                 | adopt             | carbon_capture         |         0.318 |        0.299 |
| port_electrification        | Port and Terminal Electrification                   | off-domain        | nykvist_evcharging     |         0.362 |        0.169 |
| rail_freight_efficiency     | Rail Freight Energy and Capacity Efficiency         | overlaps-existing | roadfreight_metareview |         0.401 |        0.258 |
| heavy_truck_electrification | Heavy-Duty Truck Electrification and Depot Charging | overlaps-existing | nykvist_evcharging     |         0.516 |        0.202 |

### Water, waste & emissions

| theme                       | use_case_name                                    | verdict   | nearest_existing   |   nearest_cos |   domain_fit |
|:----------------------------|:-------------------------------------------------|:----------|:-------------------|--------------:|-------------:|
| ewaste_metal_recovery       | Critical Metal Recovery from Electronic Waste    | adopt     | carbon_capture     |         0.148 |        0.199 |
| pfas_treatment              | PFAS Destruction in Industrial Effluent          | adopt     | cement_binders     |         0.171 |        0.36  |
| industrial_wastewater       | Industrial Wastewater Treatment and Reuse        | adopt     | carbon_capture     |         0.192 |        0.447 |
| plastics_chemical_recycling | Chemical Recycling of Mixed Plastic Waste        | adopt     | carbon_capture     |         0.218 |        0.289 |
| dust_emissions_control      | Industrial Particulate and Dust Emission Control | adopt     | carbon_capture     |         0.249 |        0.415 |
| flue_gas_treatment          | Flue Gas Desulfurisation and NOx Control         | adopt     | carbon_capture     |         0.29  |        0.404 |

## How to re-run this

```bash
python scripts/usecase_coverage.py
```

The comparison set is whatever briefs are cached at run time, so onboarding a use case and re-running updates every gap and verdict automatically. Edit `scripts/usecase_themes.json` to add themes you are actually considering or delete ones you never would — only the new ones get embedded. Run `--model qwen4b` for a second opinion: the diversity notebook §5 found embedding models agree on the *ordering* of close pairs while disagreeing on absolute values, so a theme that changes verdict between models is one whose margin was never real.
