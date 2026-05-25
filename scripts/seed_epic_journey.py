#!/usr/bin/env python3
"""Reset and seed the DML store with a detailed interstellar expedition corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


MISSION = "Asteria Crossing"
LATTICE_ID = "asteria-crossing-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_path() -> None:
    core = _repo_root() / "dml_core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))


def _record(title: str, category: str, layer: int, text: str, *, salience: float = 0.72, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "title": title,
        "category": category,
        "layer": layer,
        "salience": salience,
        "tags": tags or [],
        "text": f"{title}\n\n{text.strip()}",
    }


def build_records() -> list[dict[str, Any]]:
    crew = [
        ("Captain Mira Solenne", "mission command", "Former Jovian rescue commander; calm under delay-heavy comms, keeps handwritten risk ledgers, and is authorized to abort the crossing if casualty projections exceed 8%."),
        ("Commander Elias Voss", "flight director", "Navigation specialist who built the adaptive burn planner; insists every course correction has a passive fallback and maintains the golden trajectory archive."),
        ("Dr. Anika Rao", "chief scientist", "Exoplanet climatologist; owns the survey priority stack for Asteria b and will trade telescope time aggressively for atmospheric isotope windows."),
        ("Tomas Ilyin", "propulsion chief", "Fusion drive custodian; knows every pump by sound, distrusts fully autonomous valve decisions, and carries the antimatter interlock codes with Captain Solenne."),
        ("Sera Okonkwo", "systems architect", "Maintains the ship mesh, memory vaults, and fault-tolerant compute; designed the shipboard agent policy that limits autonomous repair writes."),
        ("Jun Park", "quartermaster", "Inventory lead; tracks consumables at the pouch and cartridge level, favors pessimistic spoilage models, and can recut meal plans inside 30 minutes."),
        ("Dr. Mateo Velasquez", "medical officer", "Trauma surgeon and hibernation clinician; monitors bone density drift, circadian stability, and the transplant biobank."),
        ("Lina Kade", "habitat steward", "Closed-loop agriculture lead; treats the gardens as morale infrastructure as much as oxygen and calorie infrastructure."),
        ("Niko Baines", " EVA and hull chief", "Micrometeor repair lead; manages crawler drones, patch foam reserves, and the external radiator inspection cadence."),
        ("Priya Sen", "comms officer", "Laser relay operator; compresses family packets, science bursts, and governance reports through the one-way delay envelope."),
        ("Omar Halberg", "security and law", "Mediator and protocol officer; manages restricted stores, firearms lockers, and dispute arbitration during command sleep cycles."),
        ("Yvette Chandra", "education and culture", "Archivist for civilian knowledge; runs child curriculum, narrative logs, and the 16,000-item cultural library."),
        ("Noor Tejada", "robotics lead", "Maintains utility drones, surgical assistants, cargo walkers, and the autonomous refinery crawlers."),
        ("Iris Vale", "survey pilot", "Skimmer and lander pilot; trains crews for high-latency descent rehearsal and unexpected atmospheric shear."),
        ("Pavel Orlov", "materials engineer", "Owns printers, feedstock, alloy recipes, and pressure vessel patch certification."),
        ("Kei Nakamura", "psychological health", "Maintains crew cohesion metrics, conflict heat maps, and the private counseling reserve schedule."),
    ]
    ship_specs = [
        ("Hull and Spine", "ship_specs", "The generation vessel Argo Nomad is 642 meters long with a 91 meter shield cone, a 14-module pressure spine, and three counter-rotating habitation drums. Dry mass is 118,000 tonnes; departure mass is 221,000 tonnes."),
        ("Propulsion Stack", "ship_specs", "Primary thrust comes from eight D-He3 pulsed fusion chambers rated for 0.018 g continuous cruise correction and 0.19 g peak departure burns. Secondary argon plasma drives handle docking, trim, and survey insertions."),
        ("Radiation Shield", "ship_specs", "The forward shield is layered water, boron carbide, regolith ceramic, and sacrificial ice. During dust seasons, nonessential crew rotate behind the cistern decks and the ship flies shield-first."),
        ("Habitation Drums", "ship_specs", "Drum A carries command families and medical. Drum B carries agriculture, education, and fabrication. Drum C is flexible: exercise, quarantine, large assembly, and emergency shelter."),
        ("Power Plant", "ship_specs", "Two compact aneutronic fusion plants provide 340 MW electric cruise output. Superconducting flywheels bridge pulse loads for laser comms, printers, and shield field coils."),
        ("Computer Core", "ship_specs", "The ship runs three isolated compute fabrics: command, habitat, and science. Consensus writes require two fabrics unless a declared emergency grants Captain Solenne single-key override."),
        ("Life Support", "ship_specs", "Air is balanced through algae panels, amine scrubbers, ceramic oxygen candles, and garden mass. Water recovery averages 96.7%; emergency rationing assumes 91.5% recovery for 180 days."),
        ("Docking and Craft", "ship_specs", "The vessel carries four survey skimmers, two heavy landers, six tug drones, and twenty-four hull crawlers. Only one heavy lander can be rebuilt from onboard feedstock."),
    ]
    fuel = [
        ("Fuel Reserves Ledger", "fuel", "Fuel reserves at departure: 41,200 tonnes deuterium slush, 7,900 tonnes helium-3, 320 kilograms antimatter catalyst, 8,400 tonnes argon, and 19,000 tonnes shield ice usable as contingency reaction mass."),
        ("Burn Budget", "fuel", "Mission delta-v budget is split into departure 41%, midcourse 9%, braking 37%, survey insertion 5%, reserve 8%. Reserve may not drop below 5.5% before Asteria heliopause crossing."),
        ("Helium-3 Ration", "fuel", "He-3 is the limiting cruise reagent. Propulsion chief Ilyin allows no unplanned burns above 0.004 c equivalent without a fuel board vote and two independent leak checks."),
        ("Antimatter Catalyst Rules", "fuel", "Antimatter is stored in twelve magnetically isolated bottles. Any bottle drift above 0.7 microns triggers immediate dump-safe posture and freezes nonessential fusion starts."),
        ("Argon Utility Reserve", "fuel", "Argon supports attitude control, drone refueling, and survey craft. If argon falls below 61%, skimmer training flights stop and all cargo moves switch to tether transfer."),
        ("Fuel Contamination Watch", "fuel", "Deuterium tank D-14 has a known tritium impurity trend of 0.08 ppm per year. It is safe, but should be burned before D-09 and D-11 during long correction windows."),
    ]
    inventory = [
        ("Inventory Control Authority", "inventory", "Quartermaster Jun Park controls inventory for Asteria Crossing. Park tracks consumables at pouch, cartridge, pallet, and sealed-locker level; owns ration recuts, spoilage projections, cargo audit cadence, and emergency substitution rules."),
        ("Medical Stores", "inventory", "Medical inventory includes 1,200 trauma kits, 44 surgical nanofiber packs, 18 organ scaffold cartridges, 900 hibernation stabilizer vials, and 6,400 broad-spectrum antiviral courses."),
        ("Food Baseline", "inventory", "Food stores cover 1,142 people for 14.6 years without gardens. Deep stores include grain bricks, algae paste, cultured protein starter, spice vaults, and morale-only chocolate tins."),
        ("Agriculture Seeds", "inventory", "Seed vault contains 3,200 cultivars: dwarf wheat, amaranth, soybean, potato, citrus shrub, medicinal herbs, fungal protein strains, and pollinator-free fruiting variants."),
        ("Fabrication Feedstock", "inventory", "Printer stores: 420 tonnes titanium powder, 300 tonnes aluminum-lithium, 90 tonnes copper, 33 tonnes graphene ribbon, 17 tonnes medical polymer, and 12 tonnes optical glass."),
        ("EVA Stores", "inventory", "EVA stores include 84 suits, 310 hard patches, 900 foam sealant bulbs, 48 tether packs, 18 rescue cocoons, and 24 radiation storm over-capes."),
        ("Drone Inventory", "inventory", "Robotics bay lists 24 hull crawlers, 16 interior utility drones, 8 surgical micro-drone trays, 6 cargo walkers, and 4 ice refinery scouts in sealed reserve."),
        ("Cultural Archive", "inventory", "Archive carries 16,000 books, 2,400 films, 900 instrument models, family histories, legal precedents, Earth ecological baselines, and language tutoring packs."),
        ("Restricted Stores", "inventory", "Restricted lockers contain anesthetics, reactor initiators, encryption roots, firearms for wildlife survey only, and governance seals. Omar Halberg audits them every 23 days."),
        ("Spare Electronics", "inventory", "Electronics reserve includes 14,000 general compute tiles, 2,100 radiation-hardened controllers, 700 optical bus nodes, and 90 command-grade secure enclaves."),
        ("Water Ledger", "inventory", "Potable and shield water totals 22,800 tonnes at departure. Daily habitat demand is 61.4 tonnes gross, 2.03 tonnes net loss under nominal recovery."),
    ]
    route = [
        ("Route Overview", "route", "Asteria Crossing targets the K-class star Asteria-9, 11.8 light years from Sol. Cruise velocity peaks at 0.071 c, followed by a 42-year braking arc."),
        ("Departure Leg", "route", "The departure leg slingshots past Jupiter's trailing Trojan depot, takes on final He-3 casks, then commits to the fusion ladder burn outside the main asteroid belt."),
        ("Interstellar Cruise", "route", "Cruise is divided into 18 watch eras. Each era has navigation, maintenance, cultural, and education goals so the ship never becomes a pure waiting machine."),
        ("Dust Corridor Epsilon", "route", "Dust Corridor Epsilon is a predicted debris filament at mission year 83. The ship must rotate shield-forward, stow radiators, and pause garden expansion."),
        ("Asteria Braking", "route", "Braking begins 42 years before arrival using a pulsed fusion reverse stack. The first 11 years are low thrust to protect aging chambers from thermal shock."),
        ("Target Planets", "route", "Asteria b is a temperate super-Earth candidate. Asteria c is an ocean sub-Neptune. Asteria d is icy, useful for volatiles and shield replenishment."),
    ]
    logs = [
        ("Mission Day 12", "mission_log", "Final Earth laser packet received cleanly. Crew vote named the first outbound watch 'Lantern'. Children in Drum B released paper birds into the airflow loop."),
        ("Mission Year 4", "mission_log", "Fusion chamber 3 showed injector flutter. Ilyin swapped two acoustic sensors, derated chamber 3 by 1.2%, and recovered schedule margin during the next trim burn."),
        ("Mission Year 17", "mission_log", "Garden citrus failed in rack B-7 due to fungal bloom. Lina Kade isolated the cultivar, saved 61% of rootstock, and added ultraviolet rest days."),
        ("Mission Year 38", "mission_log", "First generation handoff completed. Command authority now includes ship-born officers. Archive ceremony emphasized that Earth is origin, not destination."),
        ("Mission Year 83", "mission_log", "Dust Corridor Epsilon produced 212 shield impacts in 19 days. Hull crawlers patched five radiator scars; no pressure hull compromise occurred."),
        ("Mission Year 119", "mission_log", "A child named the aft maintenance tunnel 'the long cave.' Education team adopted the term in maps because crew remembered it faster during drills."),
        ("Mission Year 141", "mission_log", "Comms received a 129-year-old Sol packet confirming the Pacific climate treaty. Crew held a quiet observance; no operational changes required."),
        ("Mission Year 166", "mission_log", "Braking watch began. The old departure anthem was retired and replaced with a pulse-count ritual spoken before every reverse burn."),
    ]
    risks = [
        ("Primary Risk Register", "risk", "Top risks are He-3 loss, multi-year crop disease, command legitimacy drift, shield saturation, hibernation cascade failure, and unmodeled Asteria stellar activity."),
        ("Crop Disease Protocol", "risk", "If two staple crops fail in one watch era, Drum C converts to emergency hydroponics, cultural events move virtual, and grain brick rationing begins."),
        ("Command Legitimacy Drift", "risk", "Every generation must renew the mission charter. If approval falls below 67%, the ship convenes a slow assembly before any irreversible colony decision."),
        ("Hull Breach Drill", "risk", "Breach response uses blue lights, local pressure doors, crawler dispatch, and shelter-in-place. EVA chief Baines measures success by silence on command channels."),
        ("Medical Cascade", "risk", "Hibernation cascade means more than four simultaneous pod failures. Dr. Velasquez keeps two surgical teams awake during every long power bus maintenance window."),
        ("Asteria Unknowns", "risk", "The target system may have flare behavior outside Sol models. Science reserves include radiation kites, storm shelters, and sacrificial weather probes."),
    ]
    science = [
        ("Atmospheric Survey Goals", "science", "Asteria b priority measurements: oxygen disequilibrium, methane seasonality, cloud albedo, ocean fraction, nitrogen pressure, and industrial false-positive rejection."),
        ("Geology Payload", "science", "Lander kits include seismic beads, drill worms, isotope ovens, and sterile sample vaults. Planetary protection rules ban open-loop biology before three clean surveys."),
        ("Telescope Plan", "science", "The forward interferometer uses shield struts as baseline anchors after arrival. Rao wants 400 hours on Asteria b before any crew descent vote."),
        ("Biology Rules", "science", "If indigenous biology is detected, no settlement begins within 500 kilometers of active biospheres. Human survival does not erase contamination law."),
        ("Probe Fleet", "science", "The probe fleet has 36 atmosphere darts, 12 ocean sniffers, 8 magnetosphere sails, 6 cryobot kits, and 4 sample-return ascenders."),
    ]

    records = [
        _record("Mission Charter", "overview", 0, "The Asteria Crossing is a 208-year expedition to establish a human scientific and settlement presence in the Asteria-9 system. The charter prioritizes survival, consent across generations, biosphere protection, and preservation of technical memory. The mission is not allowed to become a conquest story; every landing decision must pass science, ethics, and resource review.", salience=0.95, tags=["charter", "mission"]),
        _record("Population Manifest", "overview", 0, "Departure population is 1,142 people: 312 command and technical crew, 488 civilian specialists, 196 children, 86 elders, and 60 rotating hibernation reserve staff. Population planning targets 1,400 to 1,650 by arrival, with education pipelines for propulsion, medicine, agronomy, governance, and survey operations.", salience=0.9, tags=["crew", "population"]),
        _record("Governance Compact", "overview", 0, "The ship uses a layered compact: captain for immediate safety, council for resource policy, assembly for generational legitimacy, and science veto for contamination risks. Votes are delayed during emergencies but must be audited once safe.", salience=0.87, tags=["governance"]),
    ]
    records.extend(_record(name, "crew", 2, f"{name} serves as {role}. {bio} Key demo retrieval hooks: ask about crew biography, role, authority, risk behavior, or operational responsibility.", salience=0.78, tags=["crew", role]) for name, role, bio in crew)
    records.extend(_record(title, "ship_specs", 1, text, salience=0.82, tags=["ship", title.lower().replace(" ", "_")]) for title, _, text in ship_specs)
    records.extend(_record(title, "fuel", 3, text, salience=0.86, tags=["fuel", "propulsion"]) for title, _, text in fuel)
    records.extend(_record(title, "inventory", 3, text, salience=0.8, tags=["inventory"]) for title, _, text in inventory)
    records.extend(_record(title, "route", 4, text, salience=0.82, tags=["route"]) for title, _, text in route)
    records.extend(_record(title, "mission_log", 4, text, salience=0.7, tags=["log"]) for title, _, text in logs)
    records.extend(_record(title, "risk", 4, text, salience=0.84, tags=["risk"]) for title, _, text in risks)
    records.extend(_record(title, "science", 4, text, salience=0.78, tags=["science"]) for title, _, text in science)

    systems = ["thermal loop", "greywater recovery", "cabin pressure", "printer farm", "laser comms", "navigation clock", "shield ice pumps", "garden pollination fans", "medical cold vault", "drone charging rail", "argon manifold", "fusion injector"]
    for idx, system in enumerate(systems):
        records.append(_record(
            f"Maintenance Playbook: {system.title()}",
            "maintenance",
            1,
            f"The {system} maintenance playbook defines inspection cadence, failure signatures, spare parts, and command escalation. Nominal inspection occurs every {7 + idx * 3} days. Red condition requires Sera Okonkwo or the owning department lead to approve autonomous repair writes. The playbook stores tool kits, expected sensor ranges, and recovery order so retrieval can answer operational questions without reading full manuals.",
            salience=0.74,
            tags=["maintenance", system],
        ))
    for idx in range(1, 13):
        records.append(_record(
            f"Cargo Pallet {idx:02d} Manifest",
            "inventory",
            3,
            f"Cargo pallet {idx:02d} is a sealed deep-storage unit for the Asteria arrival phase. It contains modular shelter ribs, sterile survey tarps, ceramic fasteners, ration buffers, teaching tablets, field batteries, and numbered tamper seals. Quartermaster Jun Park's rule is that arrival pallets are not opened during cruise unless two departments certify that a substitute cannot be printed or repurposed.",
            salience=0.68,
            tags=["cargo", "inventory", f"pallet-{idx:02d}"],
        ))
    arrival_sites = [
        ("Morrow Basin", "basalt plain with stable winds and low biological ambiguity", "shelter ceramics and landing beacon anchors"),
        ("Glassmere Delta", "ancient river fan with hydrated minerals and possible brine pockets", "sterile geology tents and sample vaults"),
        ("Cairn Plateau", "high-altitude plateau with excellent telescope seeing", "radiation kites and observatory truss kits"),
        ("Nacre Coast", "coastal shelf candidate with dense cloud cover and high science value", "ocean sniffers and weather probe pallets"),
        ("Hearth Saddle", "temperate saddle between two shielded ridges", "habitat ribs and emergency greenhouse film"),
        ("Vesper Crater", "old impact basin with exposed mantle signatures", "drill worms and seismic bead grids"),
    ]
    for idx, (site, character, kit) in enumerate(arrival_sites, start=1):
        records.append(_record(
            f"Arrival Site Candidate {idx}: {site}",
            "arrival_ops",
            4,
            f"{site} is an Asteria b arrival candidate described as a {character}. The site requires {kit}. Survey priority is set by Dr. Anika Rao, while Iris Vale owns descent rehearsal and Captain Solenne owns the final crewed landing go/no-go vote.",
            salience=0.79,
            tags=["arrival", "asteria_b", site.lower().replace(" ", "_")],
        ))
    watch_eras = [
        ("Lantern", "departure stabilization, family packet triage, and fusion ladder verification"),
        ("Keel", "habitat drum bearing replacement, child curriculum migration, and water ledger recalibration"),
        ("Orchard", "garden expansion, citrus cultivar recovery, and closed-loop protein diversification"),
        ("Parallax", "stellar navigation recalibration, interferometer rehearsal, and archive redundancy audit"),
        ("Quiet Hammer", "Dust Corridor Epsilon posture, shield inspection, and radiator stow drills"),
        ("Long Cave", "maintenance culture reset, map renaming, and tunnel rescue timing"),
        ("Low Ember", "braking stack recommissioning, pulse-count ritual adoption, and fuel board retraining"),
        ("First Dawn", "arrival survey triage, lander sterile lockout, and colony charter renewal"),
    ]
    for idx, (era, focus) in enumerate(watch_eras, start=1):
        records.append(_record(
            f"Watch Era {idx}: {era}",
            "mission_log",
            4,
            f"The {era} watch era focuses on {focus}. Each watch era stores operational goals, cultural continuity notes, training obligations, and known failure modes so future crews can retrieve context without reopening century-scale logs.",
            salience=0.73,
            tags=["watch_era", era.lower().replace(" ", "_")],
        ))
    departments = [
        ("Inventory", "Jun Park", "sealed stores, ration recuts, spoilage projections, and cargo substitution authority"),
        ("Medical", "Dr. Mateo Velasquez", "hibernation care, surgical readiness, quarantine thresholds, and biobank custody"),
        ("Propulsion", "Tomas Ilyin", "fusion chamber health, fuel board triggers, antimatter bottle drift, and burn authorization"),
        ("Habitat", "Lina Kade", "gardens, air balance, morale ecology, and crop disease response"),
        ("Comms", "Priya Sen", "laser relay scheduling, packet compression, Earth delay ethics, and governance reports"),
        ("Robotics", "Noor Tejada", "hull crawlers, surgical micro-drones, cargo walkers, and refinery scout reserves"),
        ("Materials", "Pavel Orlov", "printer feedstock, alloy certification, patch recipes, and pressure vessel repair"),
        ("Security", "Omar Halberg", "restricted lockers, mediation protocol, legal continuity, and wildlife-survey firearms custody"),
    ]
    for department, lead, authority in departments:
        records.append(_record(
            f"Department Authority: {department}",
            "crew",
            2,
            f"{lead} is the accountable lead for {department.lower()} operations. Authority includes {authority}. This record exists so questions about who owns a function retrieve a direct department-to-person mapping.",
            salience=0.81,
            tags=["authority", "crew", department.lower()],
        ))
    consumables = [
        ("CO2 Scrubber Matrix", "18,400 ceramic amine wafers", "replace after pressure drop exceeds 11% across two inspections"),
        ("Hydroponic Nutrient Salts", "220 tonnes mixed macro and trace minerals", "ration to lettuce and herbs first during morale-stress months"),
        ("Hibernation Cryofluid", "71,000 liters certified reserve", "quarantine any batch with crystallization above 0.03%"),
        ("Radiation Dosimeter Patches", "94,000 skin and suit badges", "issue double patches during dust corridor and flare posture"),
        ("Printer Binder Resin", "48 tonnes structural resin and 17 tonnes medical polymer", "do not cross-use medical polymer without Velasquez approval"),
        ("Laser Optics Cleaning Kits", "6,200 sealed optical swab packs", "Priya Sen controls release during relay alignment windows"),
        ("Emergency Oxygen Candles", "31,000 ceramic candles", "burn only under command emergency or confirmed scrubber cascade"),
        ("Survey Sterility Seals", "12,800 numbered sterile field seals", "science veto applies if any landing kit arrives with broken seals"),
    ]
    for name, quantity, rule in consumables:
        records.append(_record(
            f"Consumable Ledger: {name}",
            "inventory",
            3,
            f"{name} inventory stands at {quantity}. Handling rule: {rule}. Quartermaster Jun Park audits this ledger and records every substitution against mission reserve policy.",
            salience=0.76,
            tags=["consumable", "inventory", name.lower().replace(" ", "_")],
        ))
    failure_modes = [
        ("Fusion Injector Flutter", "propulsion", "acoustic drift in chamber injectors", "derate affected chamber, swap sensor pair, and schedule low-thrust recovery burn"),
        ("Garden Fungal Bloom", "habitat", "white-thread bloom across root mats", "isolate rack, ultraviolet rest cycle, and cut humidity by 6%"),
        ("Optical Bus Desync", "compute", "clock skew between command and habitat fabrics", "freeze autonomous writes and reestablish two-fabric consensus"),
        ("Crawler Adhesion Loss", "robotics", "hull crawler foot slip under dust scoring", "switch to tether crawl and replace pad set from EVA bay"),
        ("Hibernation Pod Cascade", "medical", "more than four simultaneous pod alarms", "wake surgical reserve and shed nonessential compute loads"),
        ("Argon Manifold Leak", "propulsion", "unexpected pressure drop during drone refill", "stop skimmer training and move cargo by tether transfer"),
        ("Greywater Biofilm Spike", "life_support", "filter colony growth above nominal range", "thermal shock loop and divert water to emergency mineral beds"),
        ("Governance Approval Drop", "governance", "mission approval below compact threshold", "convene slow assembly before irreversible colony decisions"),
    ]
    for name, category, signature, recovery in failure_modes:
        records.append(_record(
            f"Failure Mode: {name}",
            "risk",
            4,
            f"{name} belongs to {category}. Detection signature: {signature}. Recovery path: {recovery}. This memory is designed for incident and troubleshooting retrieval.",
            salience=0.82,
            tags=["failure_mode", category, name.lower().replace(" ", "_")],
        ))
    science_windows = [
        ("Oxygen Disequilibrium", "Asteria b atmosphere", "compare oxygen, methane, and seasonal cloud albedo before crew descent"),
        ("Sub-Neptune Ocean Signal", "Asteria c limb spectra", "search for water absorption, haze chemistry, and magnetosphere interactions"),
        ("Volatile Harvest Survey", "Asteria d ice belts", "rank shield replenishment sites by purity, spin, and dust hazard"),
        ("Stellar Flare Baseline", "Asteria-9 corona", "establish storm shelter timing and probe loss expectations"),
        ("Biosignature False Positive", "Asteria b nightside", "rule out volcanic methane and photochemical oxygen before settlement vote"),
        ("Seismic Habitability", "Morrow Basin", "use seismic beads to map crust stability under shelter foundations"),
    ]
    for title, target, objective in science_windows:
        records.append(_record(
            f"Science Window: {title}",
            "science",
            4,
            f"{title} targets {target}. Objective: {objective}. Dr. Anika Rao owns priority ranking and can delay landing rehearsals if the window resolves a biosphere or hazard ambiguity.",
            salience=0.78,
            tags=["science_window", target.lower().replace(" ", "_")],
        ))
    return records


def _summary(record: dict[str, Any], max_len: int = 300) -> str:
    text = " ".join(str(record["text"]).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _backup_and_clear_storage(storage_dir: Path) -> dict[str, Any]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = storage_dir / "backups" / f"pre-{LATTICE_ID}-{timestamp}"
    targets = [
        storage_dir / "dml_state.jsonl",
        storage_dir / "dml_store.json",
        storage_dir / "rag_store.json",
        storage_dir / "rag_index.faiss",
        storage_dir / "rag_meta.json",
        storage_dir / "visualizer_queue.json",
        storage_dir / "embedding_compatibility_report.json",
        storage_dir / "data" / "dml_state.jsonl",
    ]
    backed_up = []
    removed = []
    for target in targets:
        if not target.exists():
            continue
        backup_path = backup_dir / target.relative_to(storage_dir)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_path)
        target.unlink()
        backed_up.append(str(backup_path))
        removed.append(str(target))
    return {"backup_dir": str(backup_dir), "backed_up": backed_up, "removed": removed}


def _neighbors(index: int, width: int, count: int, id_map: dict[int, int]) -> list[int]:
    row, col = divmod(index, width)
    coords = [(row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)]
    ids = []
    for next_row, next_col in coords:
        next_index = next_row * width + next_col
        if 0 <= next_col < width and 0 <= next_index < count and next_index in id_map:
            ids.append(id_map[next_index])
    return ids


def seed_epic_journey(*, storage_dir: Path | None, replace_store: bool) -> dict[str, Any]:
    _ensure_import_path()
    from daystrom_dml.dml_adapter import DMLAdapter  # type: ignore

    overrides: dict[str, Any] = {}
    if storage_dir is not None:
        overrides["storage_dir"] = str(storage_dir.expanduser().resolve())
    resolved_storage = Path(overrides.get("storage_dir") or (_repo_root() / "data")).resolve()
    clear_report = _backup_and_clear_storage(resolved_storage) if replace_store else {}

    adapter = DMLAdapter(config_overrides=overrides or None, start_aging_loop=False)
    records = build_records()
    width = 14
    id_map: dict[int, int] = {}
    created = []
    now = time.time()
    try:
        for idx, record in enumerate(records):
            row, col = divmod(idx, width)
            meta = {
                "source": f"synthetic/{LATTICE_ID}/{record['category']}/{idx:03d}",
                "title": record["title"],
                "summary": _summary(record),
                "kind": "synthetic_expedition_memory",
                "mission": MISSION,
                "lattice_id": LATTICE_ID,
                "lattice_row": row,
                "lattice_col": col,
                "lattice_layer": int(record["layer"]),
                "lattice_size": width,
                "lattice_created_at": now,
                "category": record["category"],
                "tags": record["tags"],
                "no_merge": True,
            }
            embedding = adapter.embedder.embed(record["text"])
            item, merged = adapter.store.ingest(
                record["text"],
                embedding,
                salience=float(record["salience"]),
                fidelity=1.0,
                level=0,
                meta=meta,
            )
            if merged:
                raise RuntimeError(f"unexpected merge while seeding {record['title']}")
            item.meta["memory_id"] = item.id
            id_map[idx] = item.id
            created.append((idx, item, embedding))

        for idx, item, embedding in created:
            item.meta["lattice_neighbors"] = _neighbors(idx, width, len(created), id_map)
            item.meta["lattice_degree"] = len(item.meta["lattice_neighbors"])
            item.summary_of = [item.id]
            rag_meta = dict(item.meta)
            rag_meta.setdefault("memory_id", item.id)
            if adapter.persistent_rag_store is not None:
                adapter.persistent_rag_store.add(item.text, embedding, meta=rag_meta)
            adapter.rag_store.add_document(item.text, meta=rag_meta)

        adapter._persist_all()
        return {
            "status": "ok",
            "mission": MISSION,
            "lattice_id": LATTICE_ID,
            "created": len(created),
            "layers": sorted({record["layer"] for record in records}),
            "categories": sorted({record["category"] for record in records}),
            "storage_dir": str(adapter.storage_dir.resolve()),
            "persistence_path": str(getattr(adapter, "_persistence_path", "")),
            "clear": clear_report,
        }
    finally:
        adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", type=Path, default=None, help="Override DML storage directory")
    parser.add_argument("--replace-store", action="store_true", help="Back up and clear current DML/RAG storage before seeding")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive replacement")
    args = parser.parse_args()
    if args.replace_store and not args.yes:
        parser.error("--replace-store requires --yes")
    report = seed_epic_journey(storage_dir=args.storage_dir, replace_store=bool(args.replace_store))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
