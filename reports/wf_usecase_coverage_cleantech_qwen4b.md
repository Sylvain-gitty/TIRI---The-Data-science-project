# Use-case coverage — where new data would diversify the training set

Generated 2026-08-13 by `scripts/usecase_coverage.py` on `qwen4b`, under the **cleantech** anchor, against **34 existing use cases** and **79 candidate themes** from `scripts/usecase_themes.json`.

> **Candidate themes have briefs, not corpora.** Every score here is brief-space geometry standing in for where a use case's papers would land. Measured across the 34 use cases that have both, that substitution holds at Spearman 0.60 — real, but loose. Read this as a shortlist to go and acquire, not as a measurement. A `duplicate-risk` flag is a prompt to check against the real corpus, not a verdict.

## Calibration (computed at run time, not hardcoded)

| Landmark | Value | What it means |
|---|---|---|
| Domain pass mark | 0.287 | Weakest `in_domain_reference` use case (`cement_binders`). |
| Domain floor (guard rail) | 0.160 | Midpoint between your industrial and biomedical use cases. Below it is `off-domain`. This axis is a guard rail, not a ranking — see the stability note below. |
| Existing nearest-neighbour, median | 0.226 | How close a typical pair of use cases you already have sits. |
| Existing nearest-neighbour, max | 0.615 | The closest real pair (`synergy_hall_2012 / synergy_radjenovic_2013`) — the only pair ever measured to inflate a LOGO score. Nothing here should be allowed to reach it. |
| `overlaps-existing` line | p75 | Closer to something you own than 75% of your use cases are to their own nearest neighbour. |
| `duplicate-risk` line | p90 | Same at 90%. Check against the real corpus before acquiring. |

**Verdicts:** adopt 53, duplicate-risk 1, off-domain 5, overlaps-existing 20

### Where your existing use cases sit against the domain anchor

The guard rail is the **domain floor**, not the pass mark: the two `in_domain_reference` use cases only serve to locate the industrial mode, and everything in that mode passes. Use cases sitting between the floor and the pass mark are in-domain but less central than your reference pair — a fact about the current portfolio, not a problem.

| Use case | domain_fit | vs guard rail (0.160) |
|---|---|---|
| `carbon_capture` | 0.445 | in domain |
| `tech_forecasting` | 0.348 | in domain |
| `cement_binders` | 0.287 | in domain |
| `roadfreight_metareview` | 0.247 | in domain |
| `solar_leo` | 0.189 | in domain |
| `nykvist_evcharging` | 0.102 | off-domain |
| `ner` | 0.093 | off-domain |
| `soil_microbiome` | 0.076 | off-domain |
| `synergy_menon_2022` | -0.002 | off-domain |
| `synergy_radjenovic_2013` | -0.017 | off-domain |
| *(remaining 24 — all benchset biomedical/psychology collections)* | -0.146 to -0.022 | below |

## Which sectors are open ground — **this is the output to act on**

`median_overlap_pct` is how close the sector's themes typically sit to the nearest thing you already have — **lower means more new ground**. This ranking is the most stable thing the tool produces: Jasper and Qwen3-4B agree on it at Spearman **0.97**, against 91% agreement on individual verdicts and only 4/12 on the ordered portfolio below. Decide the *sector* here; pick themes within it for reasons this tool cannot see (data availability, customer demand, labelling cost).

| sector                     |   candidates |   open_ground |   median_overlap_pct |   median_domain_fit |
|:---------------------------|-------------:|--------------:|---------------------:|--------------------:|
| Manufacturing & automation |            7 |             4 |                0.206 |               0.172 |
| Process industries         |            4 |             3 |                0.265 |               0.21  |
| Measurement & accounting   |            3 |             2 |                0.294 |               0.365 |
| Materials & durability     |            5 |             4 |                0.382 |               0.197 |
| Oil, gas & subsea          |            5 |             4 |                0.382 |               0.264 |
| Circularity & materials    |            3 |             3 |                0.412 |               0.303 |
| Water, waste & emissions   |            6 |             5 |                0.412 |               0.308 |
| Metals & mining            |            8 |             8 |                0.485 |               0.207 |
| Non-CO2 emissions          |            2 |             2 |                0.632 |               0.309 |
| Industrial energy systems  |            4 |             3 |                0.647 |               0.387 |
| Energy & power             |            8 |             6 |                0.706 |               0.297 |
| Clean energy carriers      |            4 |             2 |                0.735 |               0.31  |
| Chemicals & process        |            6 |             3 |                0.765 |               0.33  |
| Transport & logistics      |            6 |             2 |                0.794 |               0.24  |
| Construction materials     |            3 |             1 |                0.824 |               0.239 |
| Carbon removal & capture   |            5 |             1 |                0.882 |               0.326 |

## An illustrative portfolio (top 12) — **not a procurement plan**

Greedy farthest-point selection: each pick is the candidate furthest from everything chosen so far, so the set spreads across *different* gaps instead of crowding into the largest one. `min_cos_to_set` is the pick's similarity to its closest neighbour at the moment it was chosen — lower means more new ground.

**Read this as one valid way to cover the open sectors, not as a ranking.** Greedy selection is chaotic: each pick reshuffles the remaining scores, so the two embedding models produce lists sharing only 4 of 12 entries while still agreeing on which sectors those entries come from.

|   pick | theme                        | sector                     | use_case_name                                     |   min_cos_to_set | nearest_existing       |   nearest_cos |   domain_fit |
|-------:|:-----------------------------|:---------------------------|:--------------------------------------------------|-----------------:|:-----------------------|--------------:|-------------:|
|      1 | additive_manufacturing_metal | Manufacturing & automation | Metal Additive Manufacturing Qualification        |            0.134 | soil_microbiome        |         0.134 |        0.174 |
|      2 | food_processing_energy       | Process industries         | Energy Efficiency in Food and Beverage Processing |            0.163 | roadfreight_metareview |         0.163 |        0.244 |
|      3 | offshore_wind_foundations    | Energy & power             | Offshore Wind Foundation Engineering              |            0.172 | nykvist_evcharging     |         0.172 |        0.17  |
|      4 | pipeline_integrity           | Oil, gas & subsea          | Pipeline Integrity and Corrosion Monitoring       |            0.173 | nykvist_evcharging     |         0.173 |        0.164 |
|      5 | lca_industrial               | Measurement & accounting   | Life Cycle Assessment of Industrial Products      |            0.183 | cement_binders         |         0.183 |        0.312 |
|      6 | mine_automation              | Metals & mining            | Autonomous Underground Mining Equipment           |            0.192 | roadfreight_metareview |         0.188 |        0.17  |
|      7 | textile_recycling            | Process industries         | Fibre-to-Fibre Textile Recycling                  |            0.209 | carbon_capture         |         0.163 |        0.22  |
|      8 | smr_nuclear                  | Energy & power             | Small Modular Reactor Deployment                  |            0.229 | tech_forecasting       |         0.216 |        0.29  |
|      9 | pfas_treatment               | Water, waste & emissions   | PFAS Destruction in Industrial Effluent           |            0.233 | carbon_capture         |         0.193 |        0.311 |
|     10 | steel_h2_dri                 | Metals & mining            | Hydrogen Direct Reduction Ironmaking              |            0.244 | carbon_capture         |         0.235 |        0.295 |
|     11 | saf_aviation                 | Transport & logistics      | Sustainable Aviation Fuel Production              |            0.259 | solar_leo              |         0.232 |        0.25  |
|     12 | tribology_lubricants         | Materials & durability     | Industrial Lubricants and Tribological Wear       |            0.269 | roadfreight_metareview |         0.201 |        0.273 |

## Every candidate, by sector

### Carbon removal & capture

| theme                  | use_case_name                                    | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:-----------------------|:-------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| biochar_carbon_removal | Biochar Production and Carbon Permanence         | adopt             | carbon_capture     |         0.282 |        0.311 |
| enhanced_weathering    | Enhanced Rock Weathering and Mineral Carbonation | overlaps-existing | carbon_capture     |         0.329 |        0.326 |
| cement_kiln_ccs        | Cement Kiln Carbon Capture                       | overlaps-existing | carbon_capture     |         0.44  |        0.381 |
| co2_concrete_curing    | CO2 Mineralisation and Curing in Concrete        | overlaps-existing | cement_binders     |         0.483 |        0.288 |
| direct_air_capture     | Direct Air Capture of CO2                        | duplicate-risk    | carbon_capture     |         0.515 |        0.463 |

### Chemicals & process

| theme                   | use_case_name                                      | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:------------------------|:---------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| chloralkali_efficiency  | Chlor-Alkali Electrolysis Efficiency               | adopt             | solar_leo          |         0.205 |        0.224 |
| process_intensification | Continuous Flow Process Intensification            | adopt             | carbon_capture     |         0.269 |        0.318 |
| solvent_recovery        | Industrial Solvent Recovery and Reuse              | adopt             | carbon_capture     |         0.276 |        0.393 |
| membrane_separation     | Industrial Membrane Gas Separation                 | overlaps-existing | carbon_capture     |         0.331 |        0.375 |
| green_ammonia           | Low-Pressure and Electrochemical Ammonia Synthesis | overlaps-existing | carbon_capture     |         0.335 |        0.245 |
| methanol_co2            | CO2-to-Methanol Catalytic Synthesis                | overlaps-existing | carbon_capture     |         0.376 |        0.342 |

### Circularity & materials

| theme                         | use_case_name                                     | verdict   | nearest_existing   |   nearest_cos |   domain_fit |
|:------------------------------|:--------------------------------------------------|:----------|:-------------------|--------------:|-------------:|
| circular_product_design       | Design for Disassembly and Remanufacturing        | adopt     | cement_binders     |         0.157 |        0.303 |
| battery_recycling             | Lithium-Ion Battery Recycling                     | adopt     | carbon_capture     |         0.215 |        0.215 |
| critical_mineral_substitution | Critical Mineral Substitution in Clean Technology | adopt     | cement_binders     |         0.288 |        0.399 |

### Clean energy carriers

| theme                       | use_case_name                                       | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:----------------------------|:----------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| biomethane_upgrading        | Anaerobic Digestion and Biomethane Upgrading        | adopt             | carbon_capture     |         0.249 |        0.245 |
| green_hydrogen_electrolysis | Water Electrolysis Stack Performance and Durability | adopt             | solar_leo          |         0.285 |        0.298 |
| methane_pyrolysis           | Methane Pyrolysis for Hydrogen and Solid Carbon     | overlaps-existing | carbon_capture     |         0.313 |        0.321 |
| efuels_synthesis            | E-Fuel Synthesis from Captured CO2                  | overlaps-existing | carbon_capture     |         0.366 |        0.357 |

### Construction materials

| theme              | use_case_name                                      | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:-------------------|:---------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| refractory_linings | High-Temperature Refractory Lining Durability      | adopt             | cement_binders     |         0.252 |        0.244 |
| concrete_recycling | Structural Reuse of Recycled Concrete Aggregate    | overlaps-existing | cement_binders     |         0.346 |        0.22  |
| calcined_clay_scm  | Calcined Clay Supplementary Cementitious Materials | overlaps-existing | cement_binders     |         0.442 |        0.239 |

### Energy & power

| theme                           | use_case_name                                       | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:--------------------------------|:----------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| offshore_wind_foundations       | Offshore Wind Foundation Engineering                | adopt             | nykvist_evcharging |         0.172 |        0.17  |
| smr_nuclear                     | Small Modular Reactor Deployment                    | adopt             | tech_forecasting   |         0.216 |        0.29  |
| hydrogen_storage_transport      | Hydrogen Storage and Pipeline Transport             | adopt             | carbon_capture     |         0.254 |        0.265 |
| grid_storage                    | Grid-Scale Long-Duration Energy Storage             | adopt             | solar_leo          |         0.277 |        0.329 |
| waste_heat_recovery             | Industrial Waste Heat Recovery                      | adopt             | carbon_capture     |         0.278 |        0.304 |
| geothermal_drilling             | Deep Geothermal Drilling Technology                 | adopt             | tech_forecasting   |         0.292 |        0.256 |
| industrial_heat_electrification | Electrification of High-Temperature Industrial Heat | overlaps-existing | carbon_capture     |         0.32  |        0.448 |
| high_temp_heat_pumps            | Industrial High-Temperature Heat Pumps              | overlaps-existing | carbon_capture     |         0.344 |        0.358 |

### Industrial energy systems

| theme                      | use_case_name                                    | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:---------------------------|:-------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| industrial_demand_response | Industrial Demand-Side Flexibility               | adopt             | nykvist_evcharging |         0.212 |        0.319 |
| industrial_symbiosis       | Industrial Symbiosis and Waste Heat Networks     | adopt             | cement_binders     |         0.232 |        0.406 |
| thermal_energy_storage     | Industrial Thermal Energy Storage                | adopt             | carbon_capture     |         0.3   |        0.396 |
| solar_thermal_industrial   | Concentrated Solar Heat for Industrial Processes | overlaps-existing | carbon_capture     |         0.32  |        0.378 |

### Manufacturing & automation

| theme                            | use_case_name                                                | verdict    | nearest_existing   |   nearest_cos |   domain_fit |
|:---------------------------------|:-------------------------------------------------------------|:-----------|:-------------------|--------------:|-------------:|
| ndt_inspection                   | Non-Destructive Testing of Welds and Castings                | off-domain | ner                |         0.103 |        0.099 |
| welding_automation               | Automated and Robotic Welding Process Control                | off-domain | solar_leo          |         0.105 |        0.101 |
| additive_manufacturing_metal     | Metal Additive Manufacturing Qualification                   | adopt      | soil_microbiome    |         0.134 |        0.174 |
| industrial_robotics              | Industrial Robot Manipulation in Unstructured Environments   | adopt      | solar_leo          |         0.15  |        0.193 |
| digital_twin_process             | Digital Twins for Process Plant Operations                   | adopt      | tech_forecasting   |         0.186 |        0.266 |
| machine_vision_qc                | Machine Vision Surface Defect Inspection                     | off-domain | ner                |         0.187 |        0.135 |
| predictive_maintenance_vibration | Vibration-Based Predictive Maintenance of Rotating Equipment | adopt      | synergy_hall_2012  |         0.194 |        0.172 |

### Materials & durability

| theme                 | use_case_name                                    | verdict    | nearest_existing       |   nearest_cos |   domain_fit |
|:----------------------|:-------------------------------------------------|:-----------|:-----------------------|--------------:|-------------:|
| corrosion_coatings    | Protective Coatings Against Industrial Corrosion | adopt      | tech_forecasting       |         0.139 |        0.197 |
| weld_fatigue          | Fatigue Life of Welded Steel Structures          | off-domain | cement_binders         |         0.147 |        0.074 |
| tribology_lubricants  | Industrial Lubricants and Tribological Wear      | adopt      | roadfreight_metareview |         0.201 |        0.273 |
| high_temp_alloys      | High-Temperature Structural Alloys               | adopt      | carbon_capture         |         0.268 |        0.178 |
| structural_composites | Fibre-Reinforced Composites for Heavy Structures | adopt      | cement_binders         |         0.276 |        0.202 |

### Measurement & accounting

| theme                   | use_case_name                                               | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:------------------------|:------------------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| lca_industrial          | Life Cycle Assessment of Industrial Products                | adopt             | cement_binders     |         0.183 |        0.312 |
| carbon_mrv              | Industrial Emissions Monitoring, Reporting and Verification | adopt             | tech_forecasting   |         0.187 |        0.365 |
| industrial_water_energy | Water and Energy Nexus in Industrial Decarbonisation        | overlaps-existing | carbon_capture     |         0.315 |        0.534 |

### Metals & mining

| theme                 | use_case_name                                           | verdict   | nearest_existing       |   nearest_cos |   domain_fit |
|:----------------------|:--------------------------------------------------------|:----------|:-----------------------|--------------:|-------------:|
| eaf_scrap_quality     | Electric Arc Furnace Scrap Contamination Control        | adopt     | carbon_capture         |         0.182 |        0.172 |
| ore_sorting_sensing   | Sensor-Based Ore Sorting                                | adopt     | solar_leo              |         0.186 |        0.198 |
| mine_automation       | Autonomous Underground Mining Equipment                 | adopt     | roadfreight_metareview |         0.188 |        0.17  |
| tailings_reprocessing | Mine Tailings Reprocessing and Stabilisation            | adopt     | carbon_capture         |         0.199 |        0.217 |
| rare_earth_separation | Rare Earth Element Separation and Recovery              | adopt     | carbon_capture         |         0.231 |        0.238 |
| steel_h2_dri          | Hydrogen Direct Reduction Ironmaking                    | adopt     | carbon_capture         |         0.235 |        0.295 |
| copper_hydromet       | Hydrometallurgical Copper Extraction from Low-Grade Ore | adopt     | carbon_capture         |         0.24  |        0.172 |
| aluminium_inert_anode | Inert Anode Aluminium Smelting                          | adopt     | carbon_capture         |         0.251 |        0.251 |

### Non-CO2 emissions

| theme          | use_case_name                                  | verdict   | nearest_existing   |   nearest_cos |   domain_fit |
|:---------------|:-----------------------------------------------|:----------|:-------------------|--------------:|-------------:|
| fgas_abatement | SF6 and Fluorinated Gas Alternatives           | adopt     | cement_binders     |         0.235 |        0.342 |
| n2o_abatement  | Nitrous Oxide Abatement in Chemical Production | adopt     | carbon_capture     |         0.256 |        0.276 |

### Oil, gas & subsea

| theme                  | use_case_name                                 | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:-----------------------|:----------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| subsea_inspection      | Subsea Asset Inspection Robotics              | adopt             | carbon_capture     |         0.154 |        0.175 |
| pipeline_integrity     | Pipeline Integrity and Corrosion Monitoring   | adopt             | nykvist_evcharging |         0.173 |        0.164 |
| refinery_catalysts     | Refinery Hydroprocessing Catalyst Performance | adopt             | carbon_capture     |         0.203 |        0.264 |
| methane_leak_detection | Methane Leak Detection and Quantification     | adopt             | carbon_capture     |         0.216 |        0.285 |
| flare_reduction        | Flare Gas Reduction and Utilisation           | overlaps-existing | carbon_capture     |         0.334 |        0.355 |

### Process industries

| theme                           | use_case_name                                     | verdict    | nearest_existing       |   nearest_cos |   domain_fit |
|:--------------------------------|:--------------------------------------------------|:-----------|:-----------------------|--------------:|-------------:|
| pharma_continuous_manufacturing | Continuous Pharmaceutical Manufacturing           | off-domain | synergy_chou_2003      |         0.121 |        0.117 |
| textile_recycling               | Fibre-to-Fibre Textile Recycling                  | adopt      | carbon_capture         |         0.163 |        0.22  |
| food_processing_energy          | Energy Efficiency in Food and Beverage Processing | adopt      | roadfreight_metareview |         0.163 |        0.244 |
| lignin_valorisation             | Lignin Valorisation from Pulp and Paper           | adopt      | cement_binders         |         0.192 |        0.2   |

### Transport & logistics

| theme                       | use_case_name                                       | verdict           | nearest_existing       |   nearest_cos |   domain_fit |
|:----------------------------|:----------------------------------------------------|:------------------|:-----------------------|--------------:|-------------:|
| saf_aviation                | Sustainable Aviation Fuel Production                | adopt             | solar_leo              |         0.232 |        0.25  |
| maritime_alt_fuels          | Ammonia and Methanol Marine Engines                 | adopt             | carbon_capture         |         0.262 |        0.256 |
| cold_chain_refrigeration    | Industrial Cold Chain Refrigeration                 | overlaps-existing | roadfreight_metareview |         0.309 |        0.3   |
| port_electrification        | Port and Terminal Electrification                   | overlaps-existing | nykvist_evcharging     |         0.341 |        0.212 |
| rail_freight_efficiency     | Rail Freight Energy and Capacity Efficiency         | overlaps-existing | roadfreight_metareview |         0.373 |        0.23  |
| heavy_truck_electrification | Heavy-Duty Truck Electrification and Depot Charging | overlaps-existing | nykvist_evcharging     |         0.447 |        0.201 |

### Water, waste & emissions

| theme                       | use_case_name                                    | verdict           | nearest_existing   |   nearest_cos |   domain_fit |
|:----------------------------|:-------------------------------------------------|:------------------|:-------------------|--------------:|-------------:|
| industrial_wastewater       | Industrial Wastewater Treatment and Reuse        | adopt             | carbon_capture     |         0.181 |        0.305 |
| pfas_treatment              | PFAS Destruction in Industrial Effluent          | adopt             | carbon_capture     |         0.193 |        0.311 |
| ewaste_metal_recovery       | Critical Metal Recovery from Electronic Waste    | adopt             | solar_leo          |         0.205 |        0.2   |
| plastics_chemical_recycling | Chemical Recycling of Mixed Plastic Waste        | adopt             | carbon_capture     |         0.224 |        0.252 |
| dust_emissions_control      | Industrial Particulate and Dust Emission Control | adopt             | carbon_capture     |         0.242 |        0.348 |
| flue_gas_treatment          | Flue Gas Desulfurisation and NOx Control         | overlaps-existing | carbon_capture     |         0.313 |        0.337 |

## How to re-run this

```bash
python scripts/usecase_coverage.py
```

The comparison set is whatever briefs are cached at run time, so onboarding a use case and re-running updates every gap and verdict automatically. Edit `scripts/usecase_themes.json` to add themes you are actually considering or delete ones you never would — only the new ones get embedded. Run `--model qwen4b` for a second opinion: the diversity notebook §5 found embedding models agree on the *ordering* of close pairs while disagreeing on absolute values, so a theme that changes verdict between models is one whose margin was never real.
