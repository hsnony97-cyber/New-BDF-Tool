#!/usr/bin/env python3
"""
Thickness Iteration Tool v2.0 - Advanced
==========================================
Automated thickness optimization for BDF models with property-level control.

Features:
- Reads main BDF file
- Individual property/element thickness optimization
- Automatic offset calculation and application
- Runs Nastran analysis
- Calculates RF using allowable data with power law fitting (Element & Property based)
- Sensitivity-based optimization for minimum weight
- Detailed iteration logging per folder
- Density read from BDF MAT1 cards
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import re
import threading
import csv
import shutil
import tempfile
import pandas as pd
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import numpy as np
import subprocess
import time
from datetime import datetime


class ThicknessIterationToolV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Thickness Iteration Tool v2.0 - Advanced Optimization")
        self.root.geometry("1300x950")

        # Input paths
        self.input_bdf_path = tk.StringVar()
        self.allowable_excel_path = tk.StringVar()
        self.property_excel_path = tk.StringVar()
        self.element_excel_path = tk.StringVar()
        self.nastran_path = tk.StringVar()
        self.output_folder = tk.StringVar()

        # Thickness ranges (global defaults)
        self.bar_min_thickness = tk.StringVar(value="2.0")
        self.bar_max_thickness = tk.StringVar(value="12.0")
        self.skin_min_thickness = tk.StringVar(value="3.0")
        self.skin_max_thickness = tk.StringVar(value="18.0")
        self.thickness_step = tk.StringVar(value="0.5")

        # RF settings
        self.target_rf = tk.StringVar(value="1.0")
        self.rf_tolerance = tk.StringVar(value="0.05")
        self.r2_threshold = tk.StringVar(value="0.95")
        self.min_data_points = tk.StringVar(value="3")

        # Optimization settings
        self.max_iterations = tk.StringVar(value="50")

        # Internal storage
        self.bdf_model = None

        # Property data
        self.bar_properties = {}  # PID -> {dim1, dim2, type, material}
        self.skin_properties = {}  # PID -> {thickness, material}

        # Material densities from BDF
        self.material_densities = {}  # MID -> density

        # Current thickness state (PID -> current_thickness)
        self.current_bar_thicknesses = {}
        self.current_skin_thicknesses = {}

        # Allowable fits - Property based
        self.bar_allowable_interp = {}  # PID -> {a, b, r2, excluded}
        self.skin_allowable_interp = {}  # PID -> {a, b, r2, excluded}

        # Allowable fits - Element based (like Tab 4)
        self.bar_allowable_elem_interp = {}  # EID -> {a, b, r2, excluded, property}
        self.skin_allowable_elem_interp = {}  # EID -> {a, b, r2, excluded, property}

        # Geometry data
        self.element_areas = {}  # EID -> area (shells)
        self.bar_lengths = {}  # EID -> length (bars)
        self.prop_elements = {}  # PID -> [EID list]
        self.elem_to_prop = {}  # EID -> PID
        self.prop_to_material = {}  # PID -> MID

        # Element lists for offsets
        self.landing_elem_ids = []
        self.bar_offset_elem_ids = []

        # Residual strength
        self.residual_strength_df = None

        # Results
        self.iteration_results = []
        self.best_solution = None
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI with scrolling"""
        # Main scrollable frame
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main = scrollable_frame

        # Title
        ttk.Label(main, text="Thickness Iteration Tool v2.0",
                  font=('Helvetica', 16, 'bold')).pack(pady=(10, 5))
        ttk.Label(main, text="Property/Element-level thickness optimization with RF targeting (like RF Check v2.1)",
                  font=('Helvetica', 10), foreground='gray').pack(pady=(0, 10))

        # === Section 1: Input Files ===
        self._create_input_section(main)

        # === Section 2: Thickness Ranges ===
        self._create_range_section(main)

        # === Section 3: RF Settings ===
        self._create_rf_section(main)

        # === Section 4: Optimization Settings ===
        self._create_optimization_section(main)

        # === Section 5: Actions ===
        self._create_action_section(main)

        # === Section 6: Results ===
        self._create_results_section(main)

        # === Section 7: Log ===
        self._create_log_section(main)

        # Initial instructions
        self._log_instructions()

    def _create_input_section(self, parent):
        """Create input files section"""
        frame = ttk.LabelFrame(parent, text="1. Input Files", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        # BDF File
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Main BDF File:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.input_bdf_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.input_bdf_path, "BDF Files", "*.bdf *.dat *.nas")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self.load_bdf).pack(side=tk.LEFT, padx=2)

        self.bdf_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.bdf_status.pack(anchor=tk.W, pady=2)

        # Property Excel
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Property Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.property_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.property_excel_path, "Excel", "*.xlsx *.xls")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self.load_properties).pack(side=tk.LEFT, padx=2)

        self.prop_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.prop_status.pack(anchor=tk.W, pady=2)

        # Allowable Excel
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Allowable Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.allowable_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.allowable_excel_path, "Excel/CSV", "*.xlsx *.xls *.csv")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load & Fit", command=self.load_allowable).pack(side=tk.LEFT, padx=2)

        self.allow_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.allow_status.pack(anchor=tk.W, pady=2)

        # Element IDs Excel
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Element IDs Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.element_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.element_excel_path, "Excel", "*.xlsx *.xls")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self.load_element_ids).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(Landing_Offset, Bar_Offset)", foreground='gray').pack(side=tk.LEFT, padx=5)

        self.elem_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.elem_status.pack(anchor=tk.W, pady=2)

        # Nastran Path
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Nastran Executable:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.nastran_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.nastran_path, "Executable", "*.exe *.bat *")).pack(side=tk.LEFT, padx=2)

        # Output Folder
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Output Folder:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.output_folder, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=self._browse_folder).pack(side=tk.LEFT, padx=2)

    def _create_range_section(self, parent):
        """Create thickness range section"""
        frame = ttk.LabelFrame(parent, text="2. Thickness Ranges (mm)", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        # Bar range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Bar Thickness:", width=15).pack(side=tk.LEFT)
        ttk.Label(row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.bar_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.bar_max_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="(Applied to all bar properties)", foreground='gray').pack(side=tk.LEFT, padx=10)

        # Skin range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Skin Thickness:", width=15).pack(side=tk.LEFT)
        ttk.Label(row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.skin_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.skin_max_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="(Applied to all skin properties)", foreground='gray').pack(side=tk.LEFT, padx=10)

        # Step
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Initial Step:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.thickness_step, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Step size for initial search, adaptive during optimization)", foreground='gray').pack(side=tk.LEFT, padx=10)

    def _create_rf_section(self, parent):
        """Create RF settings section"""
        frame = ttk.LabelFrame(parent, text="3. RF Target Settings", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Target RF:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.target_rf, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Tolerance (+/-):", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(row, textvariable=self.rf_tolerance, width=8).pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="R² Threshold:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.r2_threshold, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Min Data Points:", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(row, textvariable=self.min_data_points, width=8).pack(side=tk.LEFT, padx=5)

    def _create_optimization_section(self, parent):
        """Create optimization settings section"""
        frame = ttk.LabelFrame(parent, text="4. Optimization Settings", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Iterations:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.max_iterations, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Density is read from BDF MAT1 cards automatically)", foreground='gray').pack(side=tk.LEFT, padx=10)

    def _create_action_section(self, parent):
        """Create action buttons section"""
        frame = ttk.LabelFrame(parent, text="5. Actions", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)

        self.btn_start = ttk.Button(row, text=">>> START ITERATION <<<",
                                     command=self.start_iteration, width=25)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(row, text="STOP", command=self.stop_iteration,
                                    width=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Button(row, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Test Single Run", command=self.test_single_run).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(frame, text="Ready")
        self.progress_label.pack(anchor=tk.W)

    def _create_results_section(self, parent):
        """Create results section"""
        frame = ttk.LabelFrame(parent, text="6. Results Summary", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        self.result_summary = ttk.Label(frame,
                                         text="Run iteration to see results",
                                         font=('Helvetica', 11, 'bold'), foreground='blue')
        self.result_summary.pack(anchor=tk.W, pady=5)

        self.best_solution_text = tk.Text(frame, height=8, width=120, font=('Courier', 9))
        self.best_solution_text.pack(fill=tk.X, pady=5)
        self.best_solution_text.insert(tk.END, "Best solution details will appear here...")
        self.best_solution_text.config(state=tk.DISABLED)

    def _create_log_section(self, parent):
        """Create log section"""
        frame = ttk.LabelFrame(parent, text="7. Iteration Log", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        self.log_text = scrolledtext.ScrolledText(frame, height=20, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log_instructions(self):
        """Log initial instructions"""
        self.log("="*80)
        self.log("Thickness Iteration Tool v2.0 - Element/Property Based Optimization")
        self.log("="*80)
        self.log("\nWorkflow:")
        self.log("1. Load BDF file (reads elements, properties, and material densities)")
        self.log("2. Load Property Excel (Bar_Properties, Skin_Properties sheets)")
        self.log("3. Load Allowable Excel (with columns: Bar Element ID, Bar Property, d, Allowable)")
        self.log("4. Load Element IDs Excel (Landing_Offset, Bar_Offset sheets) - optional for offset")
        self.log("5. Set thickness ranges and RF target")
        self.log("6. Click 'START ITERATION'")
        self.log("\nAllowable Fitting (like RF Check v2.1):")
        self.log("- Property-based: Fits Allowable = a × T^b per Property ID")
        self.log("- Element-based: Fits Allowable = a × T^b per Element ID")
        self.log("- Uses element fit first, falls back to property fit")
        self.log("\nWeight Calculation:")
        self.log("  Skin: Sum(element_area) × thickness × density (from MAT1)")
        self.log("  Bar:  Sum(bar_length) × dim1 × dim2 × density (from MAT1)")
        self.log("="*80 + "\n")

    # ========== UTILITY FUNCTIONS ==========
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def update_progress(self, value, text=""):
        self.progress['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    def _browse_file(self, var, desc, pattern):
        f = filedialog.askopenfilename(filetypes=[(desc, pattern), ("All", "*.*")])
        if f:
            var.set(f)

    def _browse_folder(self):
        f = filedialog.askdirectory()
        if f:
            self.output_folder.set(f)

    # ========== LOADING FUNCTIONS ==========
    def load_bdf(self):
        """Load BDF and extract geometry + material densities"""
        path = self.input_bdf_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select a valid BDF file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING BDF FILE")
        self.log("="*70)

        try:
            self.bdf_model = BDF(debug=False)
            self.bdf_model.read_bdf(path, validate=False, xref=True,
                                     read_includes=True, encoding='latin-1')

            n_nodes = len(self.bdf_model.nodes)
            n_elements = len(self.bdf_model.elements)
            n_properties = len(self.bdf_model.properties)
            n_materials = len(self.bdf_model.materials)

            self.log(f"  Nodes: {n_nodes}")
            self.log(f"  Elements: {n_elements}")
            self.log(f"  Properties: {n_properties}")
            self.log(f"  Materials: {n_materials}")

            # Extract material densities
            self.material_densities = {}
            for mid, mat in self.bdf_model.materials.items():
                if hasattr(mat, 'rho') and mat.rho:
                    self.material_densities[mid] = mat.rho
                    self.log(f"    MAT1 {mid}: density = {mat.rho}")

            # Reset storage
            self.element_areas = {}
            self.bar_lengths = {}
            self.prop_elements = {}
            self.elem_to_prop = {}
            self.prop_to_material = {}

            shell_count = 0
            bar_count = 0

            # Extract property -> material mapping
            for pid, prop in self.bdf_model.properties.items():
                if hasattr(prop, 'mid') and prop.mid:
                    self.prop_to_material[pid] = prop.mid
                elif hasattr(prop, 'mid_ref') and prop.mid_ref:
                    self.prop_to_material[pid] = prop.mid_ref.mid

            for eid, elem in self.bdf_model.elements.items():
                pid = elem.pid if hasattr(elem, 'pid') else None

                if pid:
                    self.elem_to_prop[eid] = pid
                    if pid not in self.prop_elements:
                        self.prop_elements[pid] = []
                    self.prop_elements[pid].append(eid)

                if elem.type in ['CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6']:
                    shell_count += 1
                    try:
                        self.element_areas[eid] = elem.Area()
                    except:
                        self.element_areas[eid] = 0

                elif elem.type in ['CBAR', 'CBEAM']:
                    bar_count += 1
                    try:
                        self.bar_lengths[eid] = elem.Length()
                    except:
                        self.bar_lengths[eid] = 0

            self.log(f"  Shell elements: {shell_count}")
            self.log(f"  Bar elements: {bar_count}")
            self.log(f"  Total shell area: {sum(self.element_areas.values()):.2f} mm²")
            self.log(f"  Total bar length: {sum(self.bar_lengths.values()):.2f} mm")

            self.bdf_status.config(
                text=f"✓ {n_elements} elements, {n_properties} properties, {n_materials} materials",
                foreground="green"
            )

            if not self.output_folder.get():
                self.output_folder.set(os.path.dirname(path))

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.bdf_status.config(text=f"Error: {e}", foreground="red")

    def load_properties(self):
        """Load property definitions"""
        path = self.property_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select Property Excel")
            return

        self.log("\n" + "="*70)
        self.log("LOADING PROPERTY DEFINITIONS")
        self.log("="*70)

        try:
            xl = pd.ExcelFile(path)
            sheets = xl.sheet_names
            self.log(f"Sheets: {', '.join(sheets)}")

            bar_sheet = skin_sheet = residual_sheet = None
            for s in sheets:
                sl = s.lower().replace('_', '').replace(' ', '')
                if 'bar' in sl and 'prop' in sl:
                    bar_sheet = s
                elif 'skin' in sl and 'prop' in sl:
                    skin_sheet = s
                elif 'residual' in sl or 'strength' in sl:
                    residual_sheet = s

            self.bar_properties = {}
            self.skin_properties = {}
            self.current_bar_thicknesses = {}
            self.current_skin_thicknesses = {}

            bar_min = float(self.bar_min_thickness.get())
            skin_min = float(self.skin_min_thickness.get())

            if bar_sheet:
                self.log(f"\nReading '{bar_sheet}'...")
                df = pd.read_excel(xl, sheet_name=bar_sheet)
                for _, row in df.iterrows():
                    pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                    if pid:
                        dim1 = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else bar_min
                        dim2 = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else dim1
                        bar_type = str(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else 'BOX'

                        self.bar_properties[pid] = {
                            'dim1': dim1,
                            'dim2': dim2,
                            'type': bar_type
                        }
                        self.current_bar_thicknesses[pid] = bar_min

                self.log(f"  Loaded {len(self.bar_properties)} bar properties")

            if skin_sheet:
                self.log(f"\nReading '{skin_sheet}'...")
                df = pd.read_excel(xl, sheet_name=skin_sheet)
                for _, row in df.iterrows():
                    pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                    if pid:
                        t = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else skin_min
                        mat = int(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 1

                        self.skin_properties[pid] = {
                            'thickness': t,
                            'material': mat
                        }
                        self.current_skin_thicknesses[pid] = skin_min

                self.log(f"  Loaded {len(self.skin_properties)} skin properties")

            if residual_sheet:
                self.log(f"\nReading '{residual_sheet}'...")
                self.residual_strength_df = pd.read_excel(xl, sheet_name=residual_sheet)
                self.log(f"  Loaded {len(self.residual_strength_df)} residual strength rows")

            self.prop_status.config(
                text=f"✓ Bar: {len(self.bar_properties)}, Skin: {len(self.skin_properties)}",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.prop_status.config(text=f"Error", foreground="red")

    def load_allowable(self):
        """Load allowable data and fit power law curves - Element & Property based like RF Check v2.1"""
        path = self.allowable_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select Allowable file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING ALLOWABLE DATA & FITTING POWER LAW")
        self.log("(Element & Property based - like RF Check v2.1)")
        self.log("="*70)

        try:
            r2_thresh = float(self.r2_threshold.get())
            min_pts = int(self.min_data_points.get())
        except:
            r2_thresh = 0.95
            min_pts = 3

        self.log(f"R² Threshold: {r2_thresh}, Min Data Points: {min_pts}")

        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                self.log(f"\nLoaded CSV: {len(df)} rows")
                self.log(f"Columns: {list(df.columns)}")
                self._process_allowable_data(df, 'bar', r2_thresh, min_pts)
            else:
                xl = pd.ExcelFile(path)
                sheets = xl.sheet_names
                self.log(f"Sheets: {', '.join(sheets)}")

                for s in sheets:
                    sl = s.lower().replace('_', '').replace(' ', '')
                    df = pd.read_excel(xl, sheet_name=s)
                    self.log(f"\nProcessing '{s}'...")
                    self.log(f"  Columns: {list(df.columns)}")
                    self.log(f"  Rows: {len(df)}")

                    if 'bar' in sl or 'allow' in sl or 'summary' in sl:
                        self._process_allowable_data(df, 'bar', r2_thresh, min_pts)
                    elif 'skin' in sl:
                        self._process_allowable_data(df, 'skin', r2_thresh, min_pts)

            n_bar_prop = len(self.bar_allowable_interp)
            n_bar_elem = len(self.bar_allowable_elem_interp)
            n_skin_prop = len(self.skin_allowable_interp)
            n_skin_elem = len(self.skin_allowable_elem_interp)

            self.allow_status.config(
                text=f"✓ Bar: {n_bar_prop} props, {n_bar_elem} elems | Skin: {n_skin_prop} props, {n_skin_elem} elems",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.allow_status.config(text="Error", foreground="red")

    def _process_allowable_data(self, df, prop_type, r2_thresh, min_pts):
        """Process allowable data with flexible column mapping - Element & Property based"""
        self.log(f"\n  --- Processing {prop_type.upper()} allowable data ---")

        # === STEP 1: Map columns flexibly ===
        col_map = {}
        original_cols = list(df.columns)

        for col in df.columns:
            col_clean = str(col).strip()
            col_upper = col_clean.upper().replace(' ', '_').replace('-', '_')

            # Element ID mapping
            if any(x in col_upper for x in ['ELEMENT_ID', 'ELEMENT ID', 'BAR_ELEMENT_ID', 'ELEM_ID', 'EID']):
                col_map[col] = 'Element_ID'
            # Property ID mapping
            elif any(x in col_upper for x in ['PROPERTY_ID', 'PROPERTY', 'BAR_PROPERTY', 'PROP_ID', 'PID', 'BAR_PROP']):
                col_map[col] = 'Property'
            # Thickness mapping (d, t, thickness, dim)
            elif col_upper in ['D', 'T', 'THICKNESS', 'DIM', 'DIM1', 'T_MM'] or col_upper.strip() == 'D':
                col_map[col] = 'Thickness'
            # Allowable mapping
            elif any(x in col_upper for x in ['ALLOWABLE', 'ALLOW', 'ALLOWABLE_STRESS']):
                col_map[col] = 'Allowable'
            # R Value / RF
            elif any(x in col_upper for x in ['R_VALUE', 'R VALUE', 'RF', 'RVALUE']):
                col_map[col] = 'R_Value'

        self.log(f"  Column mapping:")
        for orig, mapped in col_map.items():
            self.log(f"    '{orig}' -> '{mapped}'")

        # Apply mapping
        df = df.rename(columns=col_map)

        # Check required columns
        required_property = 'Property' in df.columns
        required_thickness = 'Thickness' in df.columns
        required_allowable = 'Allowable' in df.columns
        has_element_id = 'Element_ID' in df.columns

        if not required_property:
            self.log(f"  WARNING: 'Property' column not found!")
            self.log(f"  Available columns: {list(df.columns)}")
            return

        if not required_thickness:
            self.log(f"  WARNING: 'Thickness' column not found!")
            return

        if not required_allowable:
            self.log(f"  WARNING: 'Allowable' column not found!")
            return

        self.log(f"  Has Element_ID column: {has_element_id}")

        # === STEP 2: Clean and convert data ===
        df['Property'] = pd.to_numeric(df['Property'], errors='coerce')
        df['Thickness'] = pd.to_numeric(df['Thickness'], errors='coerce')
        df['Allowable'] = pd.to_numeric(df['Allowable'], errors='coerce')

        if has_element_id:
            df['Element_ID'] = pd.to_numeric(df['Element_ID'], errors='coerce')

        # Remove invalid rows
        df = df.dropna(subset=['Property', 'Thickness', 'Allowable'])
        self.log(f"  Valid rows after cleaning: {len(df)}")

        if len(df) == 0:
            self.log("  WARNING: No valid data after cleaning!")
            return

        # === STEP 3: Property-based fitting ===
        self.log(f"\n  --- PROPERTY-BASED FITTING ---")
        prop_interp = self.bar_allowable_interp if prop_type == 'bar' else self.skin_allowable_interp

        properties = df['Property'].unique()
        self.log(f"  Unique properties: {len(properties)}")

        valid_prop_count = 0
        excluded_prop_count = 0

        for pid in properties:
            pid_int = int(pid)
            pdata = df[df['Property'] == pid]

            # For property-based: take minimum allowable per thickness
            fit_data = []
            for t in sorted(pdata['Thickness'].unique()):
                t_data = pdata[pdata['Thickness'] == t]
                min_allow = t_data['Allowable'].min()
                fit_data.append({'Thickness': float(t), 'Allowable': float(min_allow)})

            fit_df = pd.DataFrame(fit_data)
            n = len(fit_df)

            if n < min_pts:
                avg = fit_df['Allowable'].mean() if len(fit_df) > 0 else 0
                prop_interp[pid_int] = {'a': avg, 'b': 0, 'r2': 0, 'n': n, 'excluded': True}
                excluded_prop_count += 1
                continue

            result = self._fit_single_power_law(fit_df['Thickness'].values, fit_df['Allowable'].values, r2_thresh)
            result['n'] = n
            prop_interp[pid_int] = result

            if result['excluded']:
                excluded_prop_count += 1
            else:
                valid_prop_count += 1

        self.log(f"  Property fits: {valid_prop_count} valid, {excluded_prop_count} excluded")

        # Show sample property fits
        sample_count = 0
        for pid in list(prop_interp.keys())[:5]:
            p = prop_interp[pid]
            if not p['excluded']:
                self.log(f"    PID {pid}: Allowable = {p['a']:.4f} × T^({p['b']:.4f}), R²={p['r2']:.4f}")
                sample_count += 1
        if sample_count == 0:
            for pid in list(prop_interp.keys())[:3]:
                p = prop_interp[pid]
                self.log(f"    PID {pid}: Allowable = {p['a']:.4f} (constant, excluded)")

        # === STEP 4: Element-based fitting (if Element_ID exists) ===
        if has_element_id:
            self.log(f"\n  --- ELEMENT-BASED FITTING ---")
            elem_interp = self.bar_allowable_elem_interp if prop_type == 'bar' else self.skin_allowable_elem_interp

            elements = df['Element_ID'].dropna().unique()
            self.log(f"  Unique elements: {len(elements)}")

            valid_elem_count = 0
            excluded_elem_count = 0

            for eid in elements:
                eid_int = int(eid)
                edata = df[df['Element_ID'] == eid]

                # Get property for this element
                elem_pid = edata['Property'].iloc[0] if len(edata) > 0 else None
                elem_pid_int = int(elem_pid) if pd.notna(elem_pid) else None

                # For element-based: take minimum allowable per thickness
                fit_data = []
                for t in sorted(edata['Thickness'].unique()):
                    t_data = edata[edata['Thickness'] == t]
                    min_allow = t_data['Allowable'].min()
                    fit_data.append({'Thickness': float(t), 'Allowable': float(min_allow)})

                fit_df = pd.DataFrame(fit_data)
                n = len(fit_df)

                if n < min_pts:
                    avg = fit_df['Allowable'].mean() if len(fit_df) > 0 else 0
                    elem_interp[eid_int] = {'a': avg, 'b': 0, 'r2': 0, 'n': n, 'excluded': True, 'property': elem_pid_int}
                    excluded_elem_count += 1
                    continue

                result = self._fit_single_power_law(fit_df['Thickness'].values, fit_df['Allowable'].values, r2_thresh)
                result['n'] = n
                result['property'] = elem_pid_int
                elem_interp[eid_int] = result

                if result['excluded']:
                    excluded_elem_count += 1
                else:
                    valid_elem_count += 1

            self.log(f"  Element fits: {valid_elem_count} valid, {excluded_elem_count} excluded")

            # Show sample element fits
            sample_count = 0
            for eid in list(elem_interp.keys())[:5]:
                e = elem_interp[eid]
                if not e['excluded']:
                    self.log(f"    EID {eid}: Allowable = {e['a']:.4f} × T^({e['b']:.4f}), R²={e['r2']:.4f}")
                    sample_count += 1

    def _fit_single_power_law(self, x, y, r2_thresh):
        """Fit single power law curve"""
        x = np.array(x).astype(float)
        y = np.array(y).astype(float)

        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]

        if len(x) < 2:
            return {'a': np.mean(y) if len(y) > 0 else 0, 'b': 0, 'r2': 0, 'excluded': True}

        try:
            log_x, log_y = np.log(x), np.log(y)
            coeffs = np.polyfit(log_x, log_y, 1)
            b, log_a = coeffs[0], coeffs[1]
            a = np.exp(log_a)

            y_pred = a * (x ** b)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r2 < r2_thresh:
                return {'a': np.mean(y), 'b': 0, 'r2': r2, 'excluded': True}
            else:
                return {'a': a, 'b': b, 'r2': r2, 'excluded': False}

        except:
            return {'a': np.mean(y), 'b': 0, 'r2': 0, 'excluded': True}

    def load_element_ids(self):
        """Load element IDs for offset calculation"""
        path = self.element_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select Element IDs Excel")
            return

        self.log("\n" + "="*70)
        self.log("LOADING ELEMENT IDs FOR OFFSET")
        self.log("="*70)

        try:
            xl = pd.ExcelFile(path)
            sheets = xl.sheet_names
            self.log(f"Sheets: {', '.join(sheets)}")

            self.landing_elem_ids = []
            self.bar_offset_elem_ids = []

            for s in sheets:
                sl = s.lower().replace('_', '').replace(' ', '')
                if 'landing' in sl and 'offset' in sl:
                    df = pd.read_excel(xl, sheet_name=s)
                    self.landing_elem_ids = df.iloc[:, 0].dropna().astype(int).tolist()
                    self.log(f"  Landing elements: {len(self.landing_elem_ids)}")
                elif 'bar' in sl and 'offset' in sl:
                    df = pd.read_excel(xl, sheet_name=s)
                    self.bar_offset_elem_ids = df.iloc[:, 0].dropna().astype(int).tolist()
                    self.log(f"  Bar offset elements: {len(self.bar_offset_elem_ids)}")

            self.elem_status.config(
                text=f"✓ Landing: {len(self.landing_elem_ids)}, Bar: {len(self.bar_offset_elem_ids)}",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            self.elem_status.config(text="Error", foreground="red")

    # ========== HELPER FUNCTIONS ==========
    def get_allowable_stress(self, pid, thickness):
        """Get allowable stress for property at thickness using power law."""
        pid_int = int(pid) if not isinstance(pid, int) else pid

        if pid_int not in self.bar_allowable_interp:
            if pid_int not in self.skin_allowable_interp:
                return None
            params = self.skin_allowable_interp[pid_int]
        else:
            params = self.bar_allowable_interp[pid_int]

        if params.get('excluded', False) and params['b'] == 0:
            return params['a']

        return params['a'] * (thickness ** params['b'])

    def get_allowable_stress_elem(self, elem_id, thickness):
        """Get allowable stress for element at thickness using element's own power law fit."""
        elem_int = int(elem_id) if not isinstance(elem_id, int) else elem_id

        if elem_int in self.bar_allowable_elem_interp:
            params = self.bar_allowable_elem_interp[elem_int]
        elif elem_int in self.skin_allowable_elem_interp:
            params = self.skin_allowable_elem_interp[elem_int]
        else:
            return None

        if params.get('excluded', False) and params['b'] == 0:
            return params['a']

        return params['a'] * (thickness ** params['b'])

    def get_required_thickness(self, pid, target_stress, min_rf_target=1.0):
        """Calculate required thickness to achieve target RF."""
        pid_int = int(pid) if not isinstance(pid, int) else pid

        if pid_int in self.bar_allowable_interp:
            params = self.bar_allowable_interp[pid_int]
        elif pid_int in self.skin_allowable_interp:
            params = self.skin_allowable_interp[pid_int]
        else:
            return None

        a = params['a']
        b = params['b']

        if params.get('excluded', False) or b == 0:
            return None

        required_allowable = abs(target_stress) * min_rf_target

        if a <= 0:
            return None

        try:
            ratio = required_allowable / a
            if ratio <= 0:
                return None

            if b != 0:
                t_required = ratio ** (1.0 / b)
                if t_required > 0 and t_required < 1000:
                    return t_required
            return None
        except:
            return None

    def get_density_for_property(self, pid):
        """Get material density for a property from BDF"""
        if pid in self.prop_to_material:
            mid = self.prop_to_material[pid]
            if mid in self.material_densities:
                return self.material_densities[mid]

        # Default aluminum density if not found
        return 2.7e-9

    # ========== ITERATION CORE ==========
    def start_iteration(self):
        """Start optimization iteration"""
        if not self.bdf_model:
            messagebox.showerror("Error", "Load BDF first")
            return
        if not self.bar_allowable_interp and not self.skin_allowable_interp:
            messagebox.showerror("Error", "Load allowable data first")
            return
        if not self.output_folder.get():
            messagebox.showerror("Error", "Select output folder")
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.iteration_results = []
        self.best_solution = None

        threading.Thread(target=self._run_optimization, daemon=True).start()

    def stop_iteration(self):
        self.is_running = False
        self.log("\n*** STOPPING ***")

    def test_single_run(self):
        """Test a single iteration without optimization"""
        if not self.bdf_model:
            messagebox.showerror("Error", "Load BDF first")
            return

        threading.Thread(target=self._test_single, daemon=True).start()

    def _test_single(self):
        """Run a single test iteration"""
        try:
            self.log("\n" + "="*70)
            self.log("TEST SINGLE RUN")
            self.log("="*70)

            bar_min = float(self.bar_min_thickness.get())
            skin_min = float(self.skin_min_thickness.get())

            for pid in self.bar_properties:
                self.current_bar_thicknesses[pid] = bar_min
            for pid in self.skin_properties:
                self.current_skin_thicknesses[pid] = skin_min

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_folder = os.path.join(self.output_folder.get(), f"test_{timestamp}")
            os.makedirs(test_folder, exist_ok=True)

            result = self._run_single_iteration(test_folder, 0)

            if result:
                self.log(f"\nTest complete!")
                self.log(f"  Min RF: {result['min_rf']:.4f}")
                self.log(f"  Weight: {result['weight']:.6f} tonnes")
                self.log(f"  Failures: {result['n_fail']}")
                self.log(f"  Folder: {test_folder}")

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _run_optimization(self):
        """Main optimization loop"""
        try:
            self.log("\n" + "="*70)
            self.log("STARTING OPTIMIZATION")
            self.log("="*70)

            bar_min = float(self.bar_min_thickness.get())
            bar_max = float(self.bar_max_thickness.get())
            skin_min = float(self.skin_min_thickness.get())
            skin_max = float(self.skin_max_thickness.get())
            step = float(self.thickness_step.get())
            target_rf = float(self.target_rf.get())
            rf_tol = float(self.rf_tolerance.get())
            max_iter = int(self.max_iterations.get())

            self.log(f"\nParameters:")
            self.log(f"  Bar range: {bar_min} - {bar_max} mm")
            self.log(f"  Skin range: {skin_min} - {skin_max} mm")
            self.log(f"  Target RF: {target_rf} ± {rf_tol}")
            self.log(f"  Max iterations: {max_iter}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_folder = os.path.join(self.output_folder.get(), f"optimization_{timestamp}")
            os.makedirs(base_folder, exist_ok=True)

            for pid in self.bar_properties:
                self.current_bar_thicknesses[pid] = bar_min
            for pid in self.skin_properties:
                self.current_skin_thicknesses[pid] = skin_min

            iteration = 0
            best_weight = float('inf')

            while iteration < max_iter and self.is_running:
                iteration += 1

                self.log(f"\n{'='*60}")
                self.log(f"ITERATION {iteration}")
                self.log(f"{'='*60}")

                progress = (iteration / max_iter) * 100
                self.update_progress(progress, f"Iteration {iteration}/{max_iter}")

                iter_folder = os.path.join(base_folder, f"iter_{iteration:03d}")
                os.makedirs(iter_folder, exist_ok=True)

                result = self._run_single_iteration(iter_folder, iteration)

                if not result:
                    self.log("  Iteration failed, increasing all thicknesses...")
                    for pid in self.current_bar_thicknesses:
                        self.current_bar_thicknesses[pid] = min(
                            self.current_bar_thicknesses[pid] + step, bar_max)
                    for pid in self.current_skin_thicknesses:
                        self.current_skin_thicknesses[pid] = min(
                            self.current_skin_thicknesses[pid] + step, skin_max)
                    continue

                self.iteration_results.append(result)

                if result['min_rf'] >= target_rf - rf_tol and result['weight'] < best_weight:
                    best_weight = result['weight']
                    self.best_solution = result.copy()
                    self.log(f"\n  *** NEW BEST: Weight = {best_weight:.6f} t, RF = {result['min_rf']:.4f} ***")

                if result['n_fail'] > 0:
                    self._update_thicknesses_smart(result, step, bar_min, bar_max, skin_min, skin_max, target_rf)
                else:
                    self._reduce_overdesigned(result, step, bar_min, skin_min, target_rf, rf_tol)

            # Final summary
            self.log("\n" + "="*70)
            self.log("OPTIMIZATION COMPLETE")
            self.log("="*70)

            if self.best_solution:
                self.log(f"\nBest Solution:")
                self.log(f"  Weight: {self.best_solution['weight']:.6f} tonnes")
                self.log(f"  Min RF: {self.best_solution['min_rf']:.4f}")
                self.log(f"  Folder: {self.best_solution['folder']}")

                self.root.after(0, self._update_ui_results)

            self._save_all_results(base_folder)

            self.root.after(0, lambda: messagebox.showinfo("Complete",
                f"Optimization complete!\nIterations: {iteration}\nBest weight: {best_weight:.6f} t"))

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.is_running = False
            self.root.after(0, lambda: [
                self.btn_start.config(state=tk.NORMAL),
                self.btn_stop.config(state=tk.DISABLED),
                self.update_progress(100, "Complete")
            ])

    def _run_single_iteration(self, iter_folder, iteration):
        """Run a single iteration"""
        try:
            target_rf = float(self.target_rf.get())

            self.log("  [1] Updating BDF properties...")
            bdf_path = self._write_updated_bdf(iter_folder)
            if not bdf_path:
                return None

            self.log("  [2] Applying offsets...")
            offset_bdf = self._apply_offsets(bdf_path, iter_folder)
            if not offset_bdf:
                offset_bdf = bdf_path

            self.log("  [3] Running Nastran...")
            if not self._run_nastran(offset_bdf, iter_folder):
                self.log("    WARNING: Nastran failed")

            self.log("  [4] Extracting stresses...")
            stresses = self._extract_stresses(iter_folder)

            self.log("  [5] Calculating RF (Element & Property based)...")
            rf_results = self._calculate_rf_element_property_based(stresses, target_rf)

            self.log("  [6] Calculating weight...")
            weight = self._calculate_total_weight()

            min_rf = rf_results['min_rf'] if rf_results else 0
            n_fail = rf_results['n_fail'] if rf_results else 0
            n_total = rf_results['n_total'] if rf_results else 0

            self.log(f"\n  Results: Min RF = {min_rf:.4f}, Failures = {n_fail}/{n_total}, Weight = {weight:.6f} t")

            result = {
                'iteration': iteration,
                'min_rf': min_rf,
                'mean_rf': rf_results.get('mean_rf', 0) if rf_results else 0,
                'n_fail': n_fail,
                'n_total': n_total,
                'weight': weight,
                'folder': iter_folder,
                'bar_thicknesses': self.current_bar_thicknesses.copy(),
                'skin_thicknesses': self.current_skin_thicknesses.copy(),
                'failing_pids': rf_results.get('failing_pids', set()) if rf_results else set(),
                'rf_results': rf_results
            }

            self._save_iteration_results(iter_folder, result, rf_results)

            return result

        except Exception as e:
            self.log(f"  ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def _write_updated_bdf(self, iter_folder):
        """Write BDF with updated property thicknesses"""
        try:
            input_bdf = self.input_bdf_path.get()
            output_bdf = os.path.join(iter_folder, "model.bdf")

            with open(input_bdf, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            new_lines = []
            i = 0
            bar_updated = 0
            skin_updated = 0

            while i < len(lines):
                line = lines[i]

                if line.startswith('PBARL'):
                    try:
                        pid = int(line[8:16].strip())
                        if pid in self.current_bar_thicknesses:
                            t = self.current_bar_thicknesses[pid]
                            new_lines.append(line)
                            i += 1
                            while i < len(lines) and (lines[i].startswith('+') or lines[i].startswith('*') or
                                  (lines[i].startswith(' ') and lines[i].strip())):
                                cont = lines[i]
                                if cont.strip() and not cont.strip().startswith('$'):
                                    new_cont = cont[:8] + f"{t:8.4f}" + f"{t:8.4f}"
                                    if len(cont) > 24:
                                        new_cont += cont[24:]
                                    else:
                                        new_cont += '\n'
                                    new_lines.append(new_cont)
                                    bar_updated += 1
                                else:
                                    new_lines.append(cont)
                                i += 1
                            continue
                    except:
                        pass

                elif line.startswith('PSHELL'):
                    try:
                        pid = int(line[8:16].strip())
                        if pid in self.current_skin_thicknesses:
                            t = self.current_skin_thicknesses[pid]
                            new_line = line[:24] + f"{t:8.4f}" + line[32:]
                            new_lines.append(new_line)
                            skin_updated += 1
                            i += 1
                            continue
                    except:
                        pass

                new_lines.append(line)
                i += 1

            with open(output_bdf, 'w', encoding='latin-1') as f:
                f.writelines(new_lines)

            self.log(f"    Updated: {bar_updated} PBARL, {skin_updated} PSHELL")
            return output_bdf

        except Exception as e:
            self.log(f"    ERROR writing BDF: {e}")
            return None

    def _apply_offsets(self, bdf_path, iter_folder):
        """Calculate and apply offsets"""
        if not self.landing_elem_ids and not self.bar_offset_elem_ids:
            return bdf_path

        try:
            bdf = BDF(debug=False)
            bdf.read_bdf(bdf_path, validate=False, xref=True, read_includes=True, encoding='latin-1')

            landing_offsets = {}
            landing_normals = {}

            for eid in self.landing_elem_ids:
                if eid not in bdf.elements:
                    continue
                elem = bdf.elements[eid]
                pid = elem.pid if hasattr(elem, 'pid') else None

                if pid and pid in self.current_skin_thicknesses:
                    t = self.current_skin_thicknesses[pid]
                else:
                    t = float(self.skin_min_thickness.get())

                landing_offsets[eid] = -t / 2.0

                if elem.type in ['CQUAD4', 'CTRIA3']:
                    try:
                        nids = elem.node_ids[:3]
                        nodes = [bdf.nodes[n] for n in nids if n in bdf.nodes]
                        if len(nodes) >= 3:
                            p1 = np.array(nodes[0].get_position())
                            p2 = np.array(nodes[1].get_position())
                            p3 = np.array(nodes[2].get_position())
                            normal = np.cross(p2-p1, p3-p1)
                            norm_len = np.linalg.norm(normal)
                            if norm_len > 1e-10:
                                landing_normals[eid] = normal / norm_len
                    except:
                        pass

            node_to_shells = {}
            for eid, elem in bdf.elements.items():
                if elem.type in ['CQUAD4', 'CTRIA3']:
                    for nid in elem.node_ids:
                        if nid not in node_to_shells:
                            node_to_shells[nid] = []
                        node_to_shells[nid].append(eid)

            bar_offsets = {}
            for eid in self.bar_offset_elem_ids:
                if eid not in bdf.elements:
                    continue
                elem = bdf.elements[eid]
                if elem.type != 'CBAR':
                    continue

                pid = elem.pid if hasattr(elem, 'pid') else None
                if pid and pid in self.current_bar_thicknesses:
                    bar_t = self.current_bar_thicknesses[pid]
                else:
                    bar_t = float(self.bar_min_thickness.get())

                bar_nodes = elem.node_ids[:2]
                if bar_nodes[0] in node_to_shells and bar_nodes[1] in node_to_shells:
                    common = set(node_to_shells[bar_nodes[0]]) & set(node_to_shells[bar_nodes[1]])

                    max_t = 0
                    best_normal = None
                    for shell_eid in common:
                        if shell_eid in landing_offsets:
                            shell_pid = bdf.elements[shell_eid].pid
                            shell_t = self.current_skin_thicknesses.get(shell_pid, 0)
                            if shell_t > max_t:
                                max_t = shell_t
                                if shell_eid in landing_normals:
                                    best_normal = landing_normals[shell_eid]

                    if best_normal is not None and max_t > 0:
                        offset_mag = max_t + bar_t / 2.0
                        offset_vec = -best_normal * offset_mag
                        bar_offsets[eid] = tuple(offset_vec)

            with open(bdf_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            def fmt(v, w=8):
                s = f"{v:.4f}"
                if len(s) > w:
                    s = f"{v:.2E}"
                return s[:w].ljust(w)

            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]

                if line.startswith('CQUAD4'):
                    try:
                        eid = int(line[8:16].strip())
                        if eid in landing_offsets:
                            zoff = landing_offsets[eid]
                            new_line = line[:64] + fmt(zoff) + line[72:] if len(line) > 72 else line[:64] + fmt(zoff) + '\n'
                            new_lines.append(new_line)
                            i += 1
                            continue
                    except:
                        pass

                elif line.startswith('CBAR'):
                    try:
                        eid = int(line[8:16].strip())
                        if eid in bar_offsets:
                            vec = bar_offsets[eid]
                            new_lines.append(line.rstrip() + '+CB' + str(eid)[-4:] + '\n')
                            cont = ('+CB' + str(eid)[-4:]).ljust(8) + '        ' + '        '
                            cont += fmt(vec[0]) + fmt(vec[1]) + fmt(vec[2])
                            cont += fmt(vec[0]) + fmt(vec[1]) + fmt(vec[2]) + '\n'
                            new_lines.append(cont)
                            i += 1
                            continue
                    except:
                        pass

                new_lines.append(line)
                i += 1

            output_bdf = os.path.join(iter_folder, "model_offset.bdf")
            with open(output_bdf, 'w', encoding='latin-1') as f:
                f.writelines(new_lines)

            self.log(f"    Applied: {len(landing_offsets)} landing, {len(bar_offsets)} bar offsets")
            return output_bdf

        except Exception as e:
            self.log(f"    Offset error: {e}")
            return bdf_path

    def _run_nastran(self, bdf_path, iter_folder):
        """Run Nastran analysis"""
        nastran = self.nastran_path.get()
        if not nastran or not os.path.exists(nastran):
            self.log("    No Nastran executable")
            return False

        try:
            cmd = f'"{nastran}" "{bdf_path}" out="{iter_folder}" scratch=yes batch=no'
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.wait(timeout=600)

            op2_files = [f for f in os.listdir(iter_folder) if f.lower().endswith('.op2')]
            self.log(f"    Nastran complete: {len(op2_files)} OP2 files")
            return len(op2_files) > 0

        except subprocess.TimeoutExpired:
            self.log("    Nastran timeout")
            return False
        except Exception as e:
            self.log(f"    Nastran error: {e}")
            return False

    def _extract_stresses(self, iter_folder):
        """Extract stresses from OP2"""
        try:
            op2_files = [os.path.join(iter_folder, f) for f in os.listdir(iter_folder)
                         if f.lower().endswith('.op2')]

            if not op2_files:
                return []

            results = []
            for op2_path in op2_files:
                try:
                    op2 = OP2(debug=False)
                    op2.read_op2(op2_path)

                    if hasattr(op2, 'cbar_stress') and op2.cbar_stress:
                        for sc_id, data in op2.cbar_stress.items():
                            for i, eid in enumerate(data.element):
                                stress = data.data[0, i, 0] if len(data.data.shape) == 3 else data.data[i, 0]
                                results.append({
                                    'eid': int(eid),
                                    'type': 'bar',
                                    'stress': float(stress),
                                    'subcase': sc_id
                                })

                    if hasattr(op2, 'cquad4_stress') and op2.cquad4_stress:
                        for sc_id, data in op2.cquad4_stress.items():
                            for i, eid in enumerate(data.element):
                                stress = data.data[0, i, -1] if len(data.data.shape) == 3 else data.data[i, -1]
                                results.append({
                                    'eid': int(eid),
                                    'type': 'shell',
                                    'stress': float(stress),
                                    'subcase': sc_id
                                })

                except Exception as e:
                    self.log(f"    Warning reading OP2: {e}")

            self.log(f"    Extracted {len(results)} stress results")
            return results

        except Exception as e:
            self.log(f"    Stress extraction error: {e}")
            return []

    def _calculate_rf_element_property_based(self, stresses, target_rf):
        """Calculate RF using element fit first, then property fit (like RF Check v2.1)"""
        if not stresses:
            return {'min_rf': 0, 'mean_rf': 0, 'n_fail': 0, 'n_total': 0, 'failing_pids': set(), 'details': []}

        rf_list = []
        failing_pids = set()
        elem_fit_count = 0
        prop_fit_count = 0

        for s in stresses:
            eid = s['eid']
            stress = abs(s['stress'])
            etype = s['type']

            pid = self.elem_to_prop.get(eid)

            # Get thickness
            if etype == 'bar' and pid:
                thickness = self.current_bar_thicknesses.get(pid, float(self.bar_min_thickness.get()))
            else:
                thickness = self.current_skin_thicknesses.get(pid, float(self.skin_min_thickness.get())) if pid else float(self.skin_min_thickness.get())

            # Try element fit first, then property fit (like RF Check v2.1)
            allowable = None
            fit_source = "none"

            # Try element-based fit first
            if eid in self.bar_allowable_elem_interp:
                allowable = self.get_allowable_stress_elem(eid, thickness)
                if allowable is not None:
                    fit_source = "element"
                    elem_fit_count += 1
            elif eid in self.skin_allowable_elem_interp:
                allowable = self.get_allowable_stress_elem(eid, thickness)
                if allowable is not None:
                    fit_source = "element"
                    elem_fit_count += 1

            # Fall back to property-based fit
            if allowable is None and pid:
                allowable = self.get_allowable_stress(pid, thickness)
                if allowable is not None:
                    fit_source = "property"
                    prop_fit_count += 1

            # Calculate RF
            if stress == 0:
                rf = 999.0
                status = 'PASS'
            elif allowable and allowable > 0:
                rf = allowable / stress
                status = 'PASS' if rf >= target_rf else 'FAIL'
            else:
                rf = 0
                status = 'NO_ALLOW'

            if status == 'FAIL' and pid:
                failing_pids.add(pid)

            # Calculate required thickness
            required_thickness = None
            if status == 'FAIL' and pid and stress > 0:
                required_thickness = self.get_required_thickness(pid, stress, target_rf)

            rf_list.append({
                'eid': eid,
                'pid': pid,
                'type': etype,
                'thickness': thickness,
                'stress': stress,
                'allowable': allowable,
                'rf': rf,
                'status': status,
                'fit_source': fit_source,
                'required_thickness': required_thickness
            })

        valid_rf = [r['rf'] for r in rf_list if r['rf'] < 999 and r['rf'] > 0]
        min_rf = min(valid_rf) if valid_rf else 0
        mean_rf = np.mean(valid_rf) if valid_rf else 0
        n_fail = sum(1 for r in rf_list if r['status'] == 'FAIL')

        self.log(f"    Fit sources: {elem_fit_count} element, {prop_fit_count} property")

        return {
            'min_rf': min_rf,
            'mean_rf': mean_rf,
            'n_fail': n_fail,
            'n_total': len(rf_list),
            'failing_pids': failing_pids,
            'details': rf_list
        }

    def _calculate_total_weight(self):
        """Calculate total weight using density from BDF materials"""
        weight = 0.0

        # Shell weight
        for pid in self.skin_properties:
            t = self.current_skin_thicknesses.get(pid, 0)
            density = self.get_density_for_property(pid)
            if pid in self.prop_elements:
                area = sum(self.element_areas.get(eid, 0) for eid in self.prop_elements[pid])
                weight += area * t * density

        # Bar weight
        for pid in self.bar_properties:
            t = self.current_bar_thicknesses.get(pid, 0)
            density = self.get_density_for_property(pid)
            if pid in self.prop_elements:
                length = sum(self.bar_lengths.get(eid, 0) for eid in self.prop_elements[pid])
                weight += length * t * t * density

        return weight

    def _update_thicknesses_smart(self, result, step, bar_min, bar_max, skin_min, skin_max, target_rf):
        """Smart thickness update based on failures and required thickness"""
        failing_pids = result.get('failing_pids', set())
        rf_details = result.get('rf_results', {}).get('details', [])

        if not failing_pids:
            return

        self.log(f"  Updating {len(failing_pids)} failing properties...")

        # Calculate required thickness from RF results
        pid_required = {}
        for r in rf_details:
            if r['status'] == 'FAIL' and r.get('required_thickness'):
                pid = r['pid']
                req_t = r['required_thickness']
                if pid not in pid_required or req_t > pid_required[pid]:
                    pid_required[pid] = req_t

        for pid in failing_pids:
            if pid in self.current_bar_thicknesses:
                current = self.current_bar_thicknesses[pid]
                if pid in pid_required:
                    new_t = min(pid_required[pid] * 1.05, bar_max)  # 5% margin
                else:
                    new_t = min(current + step, bar_max)
                self.current_bar_thicknesses[pid] = max(new_t, current)

            if pid in self.current_skin_thicknesses:
                current = self.current_skin_thicknesses[pid]
                if pid in pid_required:
                    new_t = min(pid_required[pid] * 1.05, skin_max)
                else:
                    new_t = min(current + step, skin_max)
                self.current_skin_thicknesses[pid] = max(new_t, current)

    def _reduce_overdesigned(self, result, step, bar_min, skin_min, target_rf, rf_tol):
        """Try to reduce thickness for over-designed properties"""
        rf_details = result.get('rf_results', {}).get('details', [])
        if not rf_details:
            return

        pid_rf = {}
        for r in rf_details:
            pid = r.get('pid')
            rf = r.get('rf', 0)
            if pid and rf < 999:
                if pid not in pid_rf or rf < pid_rf[pid]:
                    pid_rf[pid] = rf

        reduce_threshold = target_rf + rf_tol + 0.2
        reduced = 0

        for pid, rf in pid_rf.items():
            if rf > reduce_threshold:
                if pid in self.current_bar_thicknesses:
                    current = self.current_bar_thicknesses[pid]
                    new_t = max(current - step/2, bar_min)
                    if new_t < current:
                        self.current_bar_thicknesses[pid] = new_t
                        reduced += 1

                if pid in self.current_skin_thicknesses:
                    current = self.current_skin_thicknesses[pid]
                    new_t = max(current - step/2, skin_min)
                    if new_t < current:
                        self.current_skin_thicknesses[pid] = new_t
                        reduced += 1

        if reduced > 0:
            self.log(f"  Reduced {reduced} over-designed properties")

    def _save_iteration_results(self, iter_folder, result, rf_results):
        """Save iteration results"""
        try:
            with open(os.path.join(iter_folder, "summary.csv"), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Parameter', 'Value'])
                w.writerow(['Iteration', result['iteration']])
                w.writerow(['Min_RF', result['min_rf']])
                w.writerow(['Mean_RF', result['mean_rf']])
                w.writerow(['N_Fail', result['n_fail']])
                w.writerow(['N_Total', result['n_total']])
                w.writerow(['Weight_tonnes', result['weight']])

            if rf_results and 'details' in rf_results:
                pd.DataFrame(rf_results['details']).to_csv(
                    os.path.join(iter_folder, "rf_details.csv"), index=False)

            bar_data = [{'PID': pid, 'Thickness': t} for pid, t in result['bar_thicknesses'].items()]
            skin_data = [{'PID': pid, 'Thickness': t} for pid, t in result['skin_thicknesses'].items()]

            if bar_data:
                pd.DataFrame(bar_data).to_csv(os.path.join(iter_folder, "bar_thicknesses.csv"), index=False)
            if skin_data:
                pd.DataFrame(skin_data).to_csv(os.path.join(iter_folder, "skin_thicknesses.csv"), index=False)

        except Exception as e:
            self.log(f"  Warning saving results: {e}")

    def _save_all_results(self, base_folder):
        """Save all optimization results"""
        try:
            history_data = []
            for r in self.iteration_results:
                history_data.append({
                    'iteration': r['iteration'],
                    'min_rf': r['min_rf'],
                    'mean_rf': r['mean_rf'],
                    'n_fail': r['n_fail'],
                    'weight': r['weight']
                })
            pd.DataFrame(history_data).to_csv(os.path.join(base_folder, "iteration_history.csv"), index=False)

            if self.best_solution:
                pd.DataFrame([{
                    'iteration': self.best_solution['iteration'],
                    'min_rf': self.best_solution['min_rf'],
                    'weight': self.best_solution['weight'],
                    'folder': self.best_solution['folder']
                }]).to_csv(os.path.join(base_folder, "best_solution.csv"), index=False)

            self.log(f"\nSaved results to {base_folder}")

        except Exception as e:
            self.log(f"Warning saving all results: {e}")

    def _update_ui_results(self):
        """Update UI with results"""
        if self.best_solution:
            self.result_summary.config(
                text=f"Best: Weight={self.best_solution['weight']:.6f}t, RF={self.best_solution['min_rf']:.4f}",
                foreground="green"
            )

            self.best_solution_text.config(state=tk.NORMAL)
            self.best_solution_text.delete(1.0, tk.END)

            text = f"Best Solution (Iteration {self.best_solution['iteration']}):\n"
            text += f"  Total Weight: {self.best_solution['weight']:.6f} tonnes\n"
            text += f"  Minimum RF: {self.best_solution['min_rf']:.4f}\n"
            text += f"  Failed Elements: {self.best_solution['n_fail']}\n"
            text += f"  Folder: {self.best_solution['folder']}\n\n"

            text += "Bar Thicknesses (sample):\n"
            for pid in list(self.best_solution['bar_thicknesses'].keys())[:5]:
                text += f"  PID {pid}: {self.best_solution['bar_thicknesses'][pid]:.2f} mm\n"

            text += "\nSkin Thicknesses (sample):\n"
            for pid in list(self.best_solution['skin_thicknesses'].keys())[:5]:
                text += f"  PID {pid}: {self.best_solution['skin_thicknesses'][pid]:.2f} mm\n"

            self.best_solution_text.insert(tk.END, text)
            self.best_solution_text.config(state=tk.DISABLED)

    def export_results(self):
        """Export results to Excel"""
        if not self.iteration_results:
            messagebox.showerror("Error", "No results to export")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_folder.get(), f"optimization_results_{timestamp}.xlsx")

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                history = [{
                    'iteration': r['iteration'],
                    'min_rf': r['min_rf'],
                    'n_fail': r['n_fail'],
                    'weight': r['weight']
                } for r in self.iteration_results]
                pd.DataFrame(history).to_excel(writer, sheet_name='History', index=False)

                if self.best_solution:
                    pd.DataFrame([self.best_solution]).to_excel(writer, sheet_name='Best', index=False)

                bar_data = [{'PID': p, 'Thickness': t} for p, t in self.current_bar_thicknesses.items()]
                skin_data = [{'PID': p, 'Thickness': t} for p, t in self.current_skin_thicknesses.items()]

                pd.DataFrame(bar_data).to_excel(writer, sheet_name='Bar_Thicknesses', index=False)
                pd.DataFrame(skin_data).to_excel(writer, sheet_name='Skin_Thicknesses', index=False)

            self.log(f"\nExported to: {path}")
            messagebox.showinfo("Export", f"Exported to:\n{path}")

        except Exception as e:
            self.log(f"Export error: {e}")
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = ThicknessIterationToolV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
