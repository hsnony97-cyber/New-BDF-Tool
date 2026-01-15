#!/usr/bin/env python3
"""
Thickness Iteration Tool v2.1
==============================
Per-property thickness optimization with RF Check v2.1 allowable fitting logic.

Features:
- RF Check v2.1 compatible allowable reading (exact same logic)
- Per-property individual thickness optimization
- MAT1/MAT8/MAT9 density support
- Sensitivity-based smart optimization
- Minimum weight objective with RF constraint
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import csv
import pandas as pd
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import numpy as np
import subprocess
from datetime import datetime


class ThicknessIterationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Thickness Iteration Tool v2.1")
        self.root.geometry("1300x950")

        # Input paths
        self.input_bdf_path = tk.StringVar()
        self.allowable_excel_path = tk.StringVar()
        self.property_excel_path = tk.StringVar()
        self.element_excel_path = tk.StringVar()
        self.nastran_path = tk.StringVar()
        self.output_folder = tk.StringVar()

        # Thickness ranges
        self.bar_min_thickness = tk.StringVar(value="2.0")
        self.bar_max_thickness = tk.StringVar(value="12.0")
        self.skin_min_thickness = tk.StringVar(value="3.0")
        self.skin_max_thickness = tk.StringVar(value="18.0")
        self.thickness_step = tk.StringVar(value="0.5")

        # RF settings
        self.target_rf = tk.StringVar(value="1.0")
        self.rf_tolerance = tk.StringVar(value="0.05")
        self.r2_threshold_var = tk.StringVar(value="0.95")
        self.min_data_points_var = tk.StringVar(value="3")

        # Optimization
        self.max_iterations = tk.StringVar(value="50")

        # Data storage
        self.bdf_model = None
        self.bar_properties = {}
        self.skin_properties = {}

        # Per-property current thicknesses
        self.current_bar_thicknesses = {}  # PID -> thickness
        self.current_skin_thicknesses = {}  # PID -> thickness

        # Material densities from BDF (MAT1, MAT8, MAT9, etc.)
        self.material_densities = {}  # MID -> density
        self.prop_to_material = {}  # PID -> MID

        # Geometry
        self.element_areas = {}
        self.bar_lengths = {}
        self.prop_elements = {}
        self.elem_to_prop = {}

        # Allowable fits (RF Check v2.1 format)
        self.allowable_interp = {}  # PID -> {a, b, r2, excluded, n_pts}
        self.allowable_elem_interp = {}  # EID -> {a, b, r2, excluded, property}
        self.allowable_df = None

        # Offset elements
        self.landing_elem_ids = []
        self.bar_offset_elem_ids = []

        # Results
        self.iteration_results = []
        self.best_solution = None
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main = scrollable_frame

        ttk.Label(main, text="Thickness Iteration Tool v2.1", font=('Helvetica', 16, 'bold')).pack(pady=10)
        ttk.Label(main, text="Per-property optimization with RF Check v2.1 allowable logic", foreground='gray').pack()

        # Section 1: Input Files
        f1 = ttk.LabelFrame(main, text="1. Input Files", padding=10)
        f1.pack(fill=tk.X, pady=5, padx=10)

        for label, var, cmd, status_name in [
            ("Main BDF:", self.input_bdf_path, self.load_bdf, "bdf_status"),
            ("Property Excel:", self.property_excel_path, self.load_properties, "prop_status"),
            ("Allowable Excel:", self.allowable_excel_path, self.rf_load_allowable, "allow_status"),
            ("Offset Element IDs:", self.element_excel_path, self.load_element_ids, "elem_status"),
        ]:
            row = ttk.Frame(f1)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=50).pack(side=tk.LEFT, padx=5)
            ttk.Button(row, text="Browse", command=lambda v=var: self.browse_file(v)).pack(side=tk.LEFT)
            ttk.Button(row, text="Load", command=cmd).pack(side=tk.LEFT, padx=2)
            setattr(self, status_name, ttk.Label(f1, text="Not loaded", foreground="gray"))
            getattr(self, status_name).pack(anchor=tk.W)

        row = ttk.Frame(f1)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Nastran Exe:", width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.nastran_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self.browse_file(self.nastran_path)).pack(side=tk.LEFT)

        row = ttk.Frame(f1)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Output Folder:", width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.output_folder, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)

        # Section 2: Thickness Ranges
        f2 = ttk.LabelFrame(main, text="2. Thickness Ranges (mm)", padding=10)
        f2.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(f2)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Bar:").pack(side=tk.LEFT)
        ttk.Label(row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.bar_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.bar_max_thickness, width=8).pack(side=tk.LEFT)

        row = ttk.Frame(f2)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Skin:").pack(side=tk.LEFT)
        ttk.Label(row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.skin_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.skin_max_thickness, width=8).pack(side=tk.LEFT)

        row = ttk.Frame(f2)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Step:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.thickness_step, width=8).pack(side=tk.LEFT, padx=5)

        # Section 3: RF Settings
        f3 = ttk.LabelFrame(main, text="3. RF Settings", padding=10)
        f3.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(f3)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Target RF:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.target_rf, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Tolerance:").pack(side=tk.LEFT, padx=10)
        ttk.Entry(row, textvariable=self.rf_tolerance, width=8).pack(side=tk.LEFT)

        row = ttk.Frame(f3)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="R² Threshold:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.r2_threshold_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Min Points:").pack(side=tk.LEFT, padx=10)
        ttk.Entry(row, textvariable=self.min_data_points_var, width=8).pack(side=tk.LEFT)

        # Section 4: Optimization
        f4 = ttk.LabelFrame(main, text="4. Optimization", padding=10)
        f4.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(f4)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Max Iterations:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.max_iterations, width=8).pack(side=tk.LEFT, padx=5)

        # Section 5: Actions
        f5 = ttk.LabelFrame(main, text="5. Actions", padding=10)
        f5.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(f5)
        row.pack(fill=tk.X, pady=5)
        self.btn_start = ttk.Button(row, text=">>> START OPTIMIZATION <<<", command=self.start_optimization, width=25)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop = ttk.Button(row, text="STOP", command=self.stop_optimization, width=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Export", command=self.export_results).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(f5, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        self.progress_label = ttk.Label(f5, text="Ready")
        self.progress_label.pack(anchor=tk.W)

        # Section 6: Results
        f6 = ttk.LabelFrame(main, text="6. Results", padding=10)
        f6.pack(fill=tk.X, pady=5, padx=10)

        self.result_summary = ttk.Label(f6, text="Run optimization to see results", font=('Helvetica', 11, 'bold'), foreground='blue')
        self.result_summary.pack(anchor=tk.W, pady=5)

        self.best_text = tk.Text(f6, height=10, font=('Courier', 9))
        self.best_text.pack(fill=tk.X, pady=5)

        # Section 7: Log
        f7 = ttk.LabelFrame(main, text="7. Log", padding=10)
        f7.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        self.log_text = scrolledtext.ScrolledText(f7, height=15, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log("="*70)
        self.log("Thickness Iteration Tool v2.1")
        self.log("Per-property optimization with RF Check v2.1 allowable logic")
        self.log("="*70)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def update_progress(self, val, txt=""):
        self.progress['value'] = val
        self.progress_label.config(text=txt)
        self.root.update_idletasks()

    def browse_file(self, var):
        f = filedialog.askopenfilename()
        if f:
            var.set(f)

    def browse_folder(self):
        f = filedialog.askdirectory()
        if f:
            self.output_folder.set(f)

    # ==================== BDF LOADING ====================
    def load_bdf(self):
        path = self.input_bdf_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select BDF file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING BDF")
        self.log("="*70)

        try:
            self.bdf_model = BDF(debug=False)
            self.bdf_model.read_bdf(path, validate=False, xref=True, read_includes=True, encoding='latin-1')

            self.log(f"  Nodes: {len(self.bdf_model.nodes)}")
            self.log(f"  Elements: {len(self.bdf_model.elements)}")
            self.log(f"  Properties: {len(self.bdf_model.properties)}")
            self.log(f"  Materials: {len(self.bdf_model.materials)}")

            # Extract material densities (MAT1, MAT2, MAT8, MAT9, etc.)
            self.material_densities = {}
            for mid, mat in self.bdf_model.materials.items():
                rho = None
                if hasattr(mat, 'rho') and mat.rho is not None:
                    rho = mat.rho
                elif hasattr(mat, 'Rho') and mat.Rho is not None:
                    rho = mat.Rho()
                if rho:
                    self.material_densities[mid] = rho
                    self.log(f"    Material {mid} ({mat.type}): density = {rho}")

            # Property -> Material mapping
            self.prop_to_material = {}
            for pid, prop in self.bdf_model.properties.items():
                mid = None
                if hasattr(prop, 'mid') and prop.mid:
                    mid = prop.mid if isinstance(prop.mid, int) else prop.mid.mid
                elif hasattr(prop, 'mid1') and prop.mid1:
                    mid = prop.mid1 if isinstance(prop.mid1, int) else prop.mid1.mid
                elif hasattr(prop, 'mid_ref') and prop.mid_ref:
                    mid = prop.mid_ref.mid
                if mid:
                    self.prop_to_material[pid] = mid

            # Element geometry
            self.element_areas = {}
            self.bar_lengths = {}
            self.prop_elements = {}
            self.elem_to_prop = {}

            shell_count = bar_count = 0

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

            self.log(f"  Shells: {shell_count}, Bars: {bar_count}")

            self.bdf_status.config(text=f"✓ {len(self.bdf_model.elements)} elements", foreground="green")

            if not self.output_folder.get():
                self.output_folder.set(os.path.dirname(path))

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.bdf_status.config(text="Error", foreground="red")

    def load_properties(self):
        path = self.property_excel_path.get()
        if not path:
            messagebox.showerror("Error", "Select Property Excel")
            return

        self.log("\n" + "="*70)
        self.log("LOADING PROPERTIES")
        self.log("="*70)

        try:
            xl = pd.ExcelFile(path)
            self.log(f"Sheets: {xl.sheet_names}")

            bar_min = float(self.bar_min_thickness.get())
            skin_min = float(self.skin_min_thickness.get())

            self.bar_properties = {}
            self.skin_properties = {}
            self.current_bar_thicknesses = {}
            self.current_skin_thicknesses = {}

            for sheet in xl.sheet_names:
                sl = sheet.lower().replace('_', '').replace(' ', '')
                df = pd.read_excel(xl, sheet_name=sheet)

                if 'bar' in sl and 'prop' in sl:
                    self.log(f"\nReading bar properties from '{sheet}'...")
                    for _, row in df.iterrows():
                        pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                        if pid:
                            self.bar_properties[pid] = {
                                'dim1': float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else bar_min,
                                'dim2': float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else bar_min,
                            }
                            self.current_bar_thicknesses[pid] = bar_min
                    self.log(f"  Loaded {len(self.bar_properties)} bar properties")

                elif 'skin' in sl and 'prop' in sl:
                    self.log(f"\nReading skin properties from '{sheet}'...")
                    for _, row in df.iterrows():
                        pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                        if pid:
                            self.skin_properties[pid] = {
                                'thickness': float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else skin_min,
                            }
                            self.current_skin_thicknesses[pid] = skin_min
                    self.log(f"  Loaded {len(self.skin_properties)} skin properties")

            self.prop_status.config(text=f"✓ Bar: {len(self.bar_properties)}, Skin: {len(self.skin_properties)}", foreground="green")

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.prop_status.config(text="Error", foreground="red")

    # ==================== RF CHECK v2.1 ALLOWABLE LOADING (EXACT COPY) ====================
    def rf_load_allowable(self):
        """Load Allowable stress data and fit power law - EXACT RF Check v2.1 logic."""
        path = self.allowable_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select a valid Allowable file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING ALLOWABLE DATA & FITTING POWER LAW")
        self.log("(RF Check v2.1 exact logic)")
        self.log("="*70)

        try:
            r2_threshold = float(self.r2_threshold_var.get())
            min_data_pts = int(self.min_data_points_var.get())
        except:
            r2_threshold = 0.95
            min_data_pts = 3

        self.log(f"R² Threshold: {r2_threshold}, Min Data Points: {min_data_pts}")

        try:
            # Load file
            if path.endswith('.csv'):
                raw_df = pd.read_csv(path)
            else:
                xl = pd.ExcelFile(path)
                bar_sheet = None
                for sheet in xl.sheet_names:
                    if 'bar' in sheet.lower() or 'allowable' in sheet.lower() or 'summary' in sheet.lower():
                        bar_sheet = sheet
                        break
                raw_df = pd.read_excel(path, sheet_name=bar_sheet if bar_sheet else 0)

            self.log(f"\nLoaded: {len(raw_df)} rows")
            self.log(f"Columns: {list(raw_df.columns)}")

            # Clean column names
            clean_cols = {}
            for col in raw_df.columns:
                clean_name = str(col).replace('\n', ' ').replace('\r', ' ').strip()
                clean_name = ' '.join(clean_name.split())
                clean_cols[col] = clean_name
            raw_df = raw_df.rename(columns=clean_cols)

            # Map column names (RF Check v2.1 exact mapping)
            # IMPORTANT: Only map ONE column to Thickness to avoid duplicate columns
            col_map = {}
            thickness_col_found = None

            for col in raw_df.columns:
                col_up = col.upper().replace(' ', '_').replace('BAR_', '').replace('(MM)', '').strip('_')

                if col_up in ['PROPERTY_ID', 'PROPERTY', 'PROP_ID']:
                    col_map[col] = 'Property'
                elif col_up in ['ELEMENT_ID', 'ELEMENT']:
                    col_map[col] = 'Element_ID'
                elif col_up in ['ELEMENT_TYPE', 'ELEMENT_TYP']:
                    col_map[col] = 'Element_Type'
                elif col_up in ['ALLOWABLE', 'ALLOW', 'ALLOWABLE_STRESS']:
                    col_map[col] = 'Allowable'

            # Find thickness column - prioritize 'd' for bar data, then 't'
            for col in raw_df.columns:
                col_up = col.upper().replace(' ', '_').replace('BAR_', '').replace('(MM)', '').strip('_')
                if col_up == 'D' and thickness_col_found is None:
                    thickness_col_found = col
                    col_map[col] = 'Thickness'
                    break

            # If no 'd' column, look for 't' or 'thickness'
            if thickness_col_found is None:
                for col in raw_df.columns:
                    col_up = col.upper().replace(' ', '_').replace('BAR_', '').replace('(MM)', '').strip('_')
                    if col_up in ['T', 'THICKNESS', 'T_MM'] and thickness_col_found is None:
                        thickness_col_found = col
                        col_map[col] = 'Thickness'
                        break

            self.log(f"\nColumn mapping:")
            for orig, mapped in col_map.items():
                self.log(f"  '{orig}' -> '{mapped}'")

            df = raw_df.rename(columns=col_map)

            # Save full df for element-based fitting
            df_full_elements = df.copy() if 'Element_ID' in df.columns else None

            # Check for NEW format with Element_Type
            if 'Element_Type' in df.columns or 'Element_ID' in df.columns:
                self.log("\nDetected format with Element_Type/Element_ID")
                df = self._process_new_allowable_format(df)

            self.allowable_df = df

            # Convert to numeric
            df['Thickness'] = pd.to_numeric(df['Thickness'], errors='coerce')
            df['Allowable'] = pd.to_numeric(df['Allowable'], errors='coerce')
            df['Property'] = pd.to_numeric(df['Property'], errors='coerce')
            df = df.dropna(subset=['Property', 'Thickness', 'Allowable'])

            properties = df['Property'].unique()
            self.log(f"\nFitting {len(properties)} properties...")

            self.allowable_interp = {}
            excluded_r2 = []
            excluded_data = []
            valid_props = []

            for pid in properties:
                pid_int = int(pid)
                prop_data = df[df['Property'] == pid]
                n_pts = len(prop_data)

                if n_pts < min_data_pts:
                    avg = prop_data['Allowable'].mean()
                    self.allowable_interp[pid_int] = {'a': avg, 'b': 0, 'n_pts': n_pts, 'r2': 0, 'excluded': True}
                    excluded_data.append((pid_int, n_pts))
                    continue

                try:
                    x = prop_data['Thickness'].values.astype(float)
                    y = prop_data['Allowable'].values.astype(float)

                    valid = (x > 0) & (y > 0)
                    x, y = x[valid], y[valid]

                    if len(x) < 2:
                        self.allowable_interp[pid_int] = {'a': np.mean(y), 'b': 0, 'n_pts': len(x), 'r2': 0, 'excluded': True}
                        excluded_data.append((pid_int, len(x)))
                        continue

                    # Power law fit
                    log_x, log_y = np.log(x), np.log(y)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    b, log_a = coeffs[0], coeffs[1]
                    a = np.exp(log_a)

                    # R²
                    y_pred = a * (x ** b)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    if r2 < r2_threshold:
                        self.allowable_interp[pid_int] = {'a': np.mean(y), 'b': 0, 'n_pts': n_pts, 'r2': r2, 'excluded': True}
                        excluded_r2.append((pid_int, r2, n_pts))
                    else:
                        self.allowable_interp[pid_int] = {'a': a, 'b': b, 'n_pts': n_pts, 'r2': r2, 'excluded': False}
                        valid_props.append(pid_int)

                except Exception as e:
                    self.allowable_interp[pid_int] = {'a': 100, 'b': 0, 'n_pts': n_pts, 'r2': 0, 'excluded': True}
                    excluded_data.append((pid_int, n_pts))

            self.log(f"\n{'='*50}")
            self.log(f"Valid fits (R² >= {r2_threshold}): {len(valid_props)}")
            self.log(f"Excluded (R² < {r2_threshold}): {len(excluded_r2)}")
            self.log(f"Excluded (data < {min_data_pts}): {len(excluded_data)}")

            if valid_props:
                self.log(f"\nSample valid fits (Property):")
                for pid in valid_props[:5]:
                    p = self.allowable_interp[pid]
                    self.log(f"  Property {pid}: Allow = {p['a']:.4f} × T^({p['b']:.4f}), R²={p['r2']:.4f}")

            # ELEMENT-BASED CURVE FITTING
            self.allowable_elem_interp = {}

            if df_full_elements is not None and 'Element_ID' in df_full_elements.columns:
                self.log(f"\n{'='*50}")
                self.log("ELEMENT-BASED CURVE FITTING")
                self.log(f"{'='*50}")

                df_elem = df_full_elements.copy()
                df_elem['Thickness'] = pd.to_numeric(df_elem['Thickness'], errors='coerce')
                df_elem['Allowable'] = pd.to_numeric(df_elem['Allowable'], errors='coerce')
                df_elem['Element_ID'] = pd.to_numeric(df_elem['Element_ID'], errors='coerce')
                df_elem['Property'] = pd.to_numeric(df_elem['Property'], errors='coerce')
                df_elem = df_elem.dropna(subset=['Element_ID', 'Thickness', 'Allowable'])

                elements = df_elem['Element_ID'].unique()
                self.log(f"Fitting {len(elements)} elements...")

                valid_elems = []

                for elem_id in elements:
                    elem_int = int(elem_id)
                    elem_data = df_elem[df_elem['Element_ID'] == elem_id].copy()

                    elem_pid = elem_data['Property'].iloc[0] if len(elem_data) > 0 else None
                    elem_pid_int = int(elem_pid) if pd.notna(elem_pid) else None

                    # Filter by Element_Type if exists
                    if 'Element_Type' in elem_data.columns and elem_data['Element_Type'].notna().any():
                        elem_data_sorted = elem_data.sort_values('Allowable', ascending=True)
                        critical_elem_type = elem_data_sorted.iloc[0]['Element_Type']
                        filtered_data = elem_data[elem_data['Element_Type'] == critical_elem_type].copy()

                        fit_data = []
                        thickness_vals = pd.Series(filtered_data['Thickness'].values).dropna().unique()
                        for t in sorted(thickness_vals):
                            t_data = filtered_data[filtered_data['Thickness'] == t]
                            if len(t_data) > 0:
                                fit_data.append({'Thickness': float(t), 'Allowable': float(t_data['Allowable'].min())})
                        fit_df = pd.DataFrame(fit_data)
                    else:
                        fit_data = []
                        thickness_vals = pd.Series(elem_data['Thickness'].values).dropna().unique()
                        for t in sorted(thickness_vals):
                            t_data = elem_data[elem_data['Thickness'] == t]
                            if len(t_data) > 0:
                                fit_data.append({'Thickness': float(t), 'Allowable': float(t_data['Allowable'].min())})
                        fit_df = pd.DataFrame(fit_data)

                    n_pts = len(fit_df)

                    if n_pts < min_data_pts:
                        avg = fit_df['Allowable'].mean() if len(fit_df) > 0 else 0
                        self.allowable_elem_interp[elem_int] = {'a': avg, 'b': 0, 'n_pts': n_pts, 'r2': 0, 'excluded': True, 'property': elem_pid_int}
                        continue

                    try:
                        x = fit_df['Thickness'].values.astype(float)
                        y = fit_df['Allowable'].values.astype(float)
                        valid_mask = (x > 0) & (y > 0)
                        x, y = x[valid_mask], y[valid_mask]

                        if len(x) < 2:
                            self.allowable_elem_interp[elem_int] = {'a': np.mean(y), 'b': 0, 'n_pts': len(x), 'r2': 0, 'excluded': True, 'property': elem_pid_int}
                            continue

                        log_x, log_y = np.log(x), np.log(y)
                        coeffs = np.polyfit(log_x, log_y, 1)
                        b, log_a = coeffs[0], coeffs[1]
                        a = np.exp(log_a)

                        y_pred = a * (x ** b)
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                        if r2 < r2_threshold:
                            self.allowable_elem_interp[elem_int] = {'a': np.mean(y), 'b': 0, 'n_pts': n_pts, 'r2': r2, 'excluded': True, 'property': elem_pid_int}
                        else:
                            self.allowable_elem_interp[elem_int] = {'a': a, 'b': b, 'n_pts': n_pts, 'r2': r2, 'excluded': False, 'property': elem_pid_int}
                            valid_elems.append(elem_int)

                    except:
                        self.allowable_elem_interp[elem_int] = {'a': 100, 'b': 0, 'n_pts': n_pts, 'r2': 0, 'excluded': True, 'property': elem_pid_int}

                self.log(f"Valid element fits: {len(valid_elems)}")

            self.allow_status.config(text=f"✓ Prop: {len(valid_props)}, Elem: {len(self.allowable_elem_interp)}", foreground="green")

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.allow_status.config(text="Error", foreground="red")

    def _process_new_allowable_format(self, df):
        """Process format with Element_Type - RF Check v2.1 logic."""
        self.log("  Processing Element_Type format...")

        result_data = []

        # Get unique properties safely
        if 'Property' not in df.columns:
            self.log("  WARNING: No 'Property' column found")
            return pd.DataFrame(result_data)

        properties = pd.Series(df['Property'].values).dropna().unique()

        for pid in properties:
            prop_df = df[df['Property'] == pid].copy()
            if len(prop_df) == 0:
                continue

            prop_df_sorted = prop_df.sort_values('Allowable', ascending=True)
            critical_row = prop_df_sorted.iloc[0]

            crit_elem_id = critical_row.get('Element_ID', None)
            crit_elem_type = critical_row.get('Element_Type', None)

            if crit_elem_id is not None and crit_elem_type is not None:
                try:
                    crit_id = int(crit_elem_id)
                    mask = (prop_df['Element_ID'].astype(float).astype(int) == crit_id) & (prop_df['Element_Type'] == crit_elem_type)
                    filtered_df = prop_df[mask].copy()
                except:
                    filtered_df = prop_df.copy()
            elif crit_elem_type is not None:
                filtered_df = prop_df[prop_df['Element_Type'] == crit_elem_type].copy()
            else:
                filtered_df = prop_df.copy()

            # Get unique thicknesses safely (avoid DataFrame.unique() issue)
            if 'Thickness' not in filtered_df.columns or len(filtered_df) == 0:
                continue

            thickness_values = pd.Series(filtered_df['Thickness'].values).dropna().unique()

            for t in sorted(thickness_values):
                t_data = filtered_df[filtered_df['Thickness'] == t]
                if len(t_data) > 0:
                    min_allow = t_data['Allowable'].min()
                    result_data.append({'Property': int(pid), 'Thickness': float(t), 'Allowable': float(min_allow)})

        self.log(f"  Processed: {len(properties)} properties -> {len(result_data)} data points")
        return pd.DataFrame(result_data)

    def load_element_ids(self):
        path = self.element_excel_path.get()
        if not path:
            return

        self.log("\n" + "="*70)
        self.log("LOADING ELEMENT IDs FOR OFFSET")
        self.log("="*70)

        try:
            xl = pd.ExcelFile(path)
            self.landing_elem_ids = []
            self.bar_offset_elem_ids = []

            for s in xl.sheet_names:
                sl = s.lower().replace('_', '').replace(' ', '')
                df = pd.read_excel(xl, sheet_name=s)
                if 'landing' in sl:
                    self.landing_elem_ids = df.iloc[:, 0].dropna().astype(int).tolist()
                    self.log(f"  Landing: {len(self.landing_elem_ids)}")
                elif 'bar' in sl and 'offset' in sl:
                    self.bar_offset_elem_ids = df.iloc[:, 0].dropna().astype(int).tolist()
                    self.log(f"  Bar offset: {len(self.bar_offset_elem_ids)}")

            self.elem_status.config(text=f"✓ Landing: {len(self.landing_elem_ids)}, Bar: {len(self.bar_offset_elem_ids)}", foreground="green")

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.elem_status.config(text="Error", foreground="red")

    # ==================== HELPER FUNCTIONS ====================
    def get_allowable_stress(self, pid, thickness):
        pid_int = int(pid)
        if pid_int not in self.allowable_interp:
            return None
        params = self.allowable_interp[pid_int]
        if params.get('excluded', False) and params['b'] == 0:
            return params['a']
        return params['a'] * (thickness ** params['b'])

    def get_allowable_stress_elem(self, elem_id, thickness):
        elem_int = int(elem_id)
        if elem_int not in self.allowable_elem_interp:
            return None
        params = self.allowable_elem_interp[elem_int]
        if params.get('excluded', False) and params['b'] == 0:
            return params['a']
        return params['a'] * (thickness ** params['b'])

    def get_required_thickness(self, pid, stress, min_rf=1.0):
        pid_int = int(pid)
        if pid_int not in self.allowable_interp:
            return None
        params = self.allowable_interp[pid_int]
        a, b = params['a'], params['b']
        if params.get('excluded', False) or b == 0:
            return None
        required_allow = abs(stress) * min_rf
        if a <= 0:
            return None
        try:
            ratio = required_allow / a
            if ratio <= 0:
                return None
            t_req = ratio ** (1.0 / b)
            return t_req if 0 < t_req < 1000 else None
        except:
            return None

    def get_density(self, pid):
        """Get density from property's material (MAT1/MAT8/MAT9/etc.)"""
        if pid in self.prop_to_material:
            mid = self.prop_to_material[pid]
            if mid in self.material_densities:
                return self.material_densities[mid]
        return 2.7e-9  # Default aluminum

    # ==================== OPTIMIZATION ====================
    def start_optimization(self):
        if not self.bdf_model:
            messagebox.showerror("Error", "Load BDF first")
            return
        if not self.allowable_interp:
            messagebox.showerror("Error", "Load allowable data first")
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.iteration_results = []
        self.best_solution = None

        threading.Thread(target=self._run_optimization, daemon=True).start()

    def stop_optimization(self):
        self.is_running = False
        self.log("\n*** STOPPING ***")

    def _run_optimization(self):
        try:
            self.log("\n" + "="*70)
            self.log("STARTING PER-PROPERTY OPTIMIZATION")
            self.log("="*70)

            bar_min = float(self.bar_min_thickness.get())
            bar_max = float(self.bar_max_thickness.get())
            skin_min = float(self.skin_min_thickness.get())
            skin_max = float(self.skin_max_thickness.get())
            step = float(self.thickness_step.get())
            target_rf = float(self.target_rf.get())
            rf_tol = float(self.rf_tolerance.get())
            max_iter = int(self.max_iterations.get())

            self.log(f"\nBar range: {bar_min}-{bar_max} mm")
            self.log(f"Skin range: {skin_min}-{skin_max} mm")
            self.log(f"Target RF: {target_rf} ± {rf_tol}")
            self.log(f"Max iterations: {max_iter}")

            # Initialize all properties to minimum
            for pid in self.bar_properties:
                self.current_bar_thicknesses[pid] = bar_min
            for pid in self.skin_properties:
                self.current_skin_thicknesses[pid] = skin_min

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_folder = os.path.join(self.output_folder.get(), f"opt_{timestamp}")
            os.makedirs(base_folder, exist_ok=True)

            iteration = 0
            best_weight = float('inf')

            while iteration < max_iter and self.is_running:
                iteration += 1
                self.log(f"\n{'='*60}")
                self.log(f"ITERATION {iteration}")
                self.log(f"{'='*60}")

                self.update_progress((iteration/max_iter)*100, f"Iteration {iteration}/{max_iter}")

                iter_folder = os.path.join(base_folder, f"iter_{iteration:03d}")
                os.makedirs(iter_folder, exist_ok=True)

                # Run iteration
                result = self._run_iteration(iter_folder, iteration, target_rf)

                if not result:
                    self.log("  Iteration failed, increasing all thicknesses...")
                    for pid in self.current_bar_thicknesses:
                        self.current_bar_thicknesses[pid] = min(self.current_bar_thicknesses[pid] + step, bar_max)
                    for pid in self.current_skin_thicknesses:
                        self.current_skin_thicknesses[pid] = min(self.current_skin_thicknesses[pid] + step, skin_max)
                    continue

                self.iteration_results.append(result)

                # Check for best
                if result['min_rf'] >= target_rf - rf_tol and result['weight'] < best_weight:
                    best_weight = result['weight']
                    self.best_solution = result.copy()
                    self.log(f"\n  *** NEW BEST: Weight={best_weight:.6f}t, RF={result['min_rf']:.4f} ***")

                # Per-property update based on RF
                self._smart_thickness_update(result, step, bar_min, bar_max, skin_min, skin_max, target_rf, rf_tol)

            # Summary
            self.log("\n" + "="*70)
            self.log("OPTIMIZATION COMPLETE")
            self.log("="*70)

            if self.best_solution:
                self.log(f"\nBest: Weight={self.best_solution['weight']:.6f}t, RF={self.best_solution['min_rf']:.4f}")
                self._update_ui()

            self._save_results(base_folder)

            self.root.after(0, lambda: messagebox.showinfo("Done", f"Complete!\nIterations: {iteration}\nBest weight: {best_weight:.6f}t"))

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())

        finally:
            self.is_running = False
            self.root.after(0, lambda: [self.btn_start.config(state=tk.NORMAL), self.btn_stop.config(state=tk.DISABLED)])

    def _run_iteration(self, folder, iteration, target_rf):
        try:
            # 1. Write BDF
            self.log("  [1] Writing BDF...")
            bdf_path = self._write_bdf(folder)

            # 2. Apply offsets
            self.log("  [2] Applying offsets...")
            offset_bdf = self._apply_offsets(bdf_path, folder)

            # 3. Run Nastran (if available)
            self.log("  [3] Running Nastran...")
            self._run_nastran(offset_bdf or bdf_path, folder)

            # 4. Extract stresses
            self.log("  [4] Extracting stresses...")
            stresses = self._extract_stresses(folder)

            # 5. Calculate RF
            self.log("  [5] Calculating RF...")
            rf_results = self._calculate_rf(stresses, target_rf)

            # 6. Calculate weight
            self.log("  [6] Calculating weight...")
            weight = self._calculate_weight()

            min_rf = rf_results['min_rf']
            n_fail = rf_results['n_fail']

            self.log(f"\n  Results: Min RF={min_rf:.4f}, Fails={n_fail}, Weight={weight:.6f}t")

            result = {
                'iteration': iteration,
                'min_rf': min_rf,
                'n_fail': n_fail,
                'weight': weight,
                'folder': folder,
                'bar_thicknesses': self.current_bar_thicknesses.copy(),
                'skin_thicknesses': self.current_skin_thicknesses.copy(),
                'rf_details': rf_results['details'],
                'failing_pids': rf_results['failing_pids']
            }

            self._save_iteration(folder, result)
            return result

        except Exception as e:
            self.log(f"  ERROR: {e}")
            return None

    def _write_bdf(self, folder):
        input_bdf = self.input_bdf_path.get()
        output_bdf = os.path.join(folder, "model.bdf")

        with open(input_bdf, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith('PBARL'):
                try:
                    pid = int(line[8:16].strip())
                    if pid in self.current_bar_thicknesses:
                        t = self.current_bar_thicknesses[pid]
                        new_lines.append(line)
                        i += 1
                        while i < len(lines) and (lines[i].startswith('+') or lines[i].startswith('*') or (lines[i].startswith(' ') and lines[i].strip())):
                            cont = lines[i]
                            if cont.strip() and not cont.strip().startswith('$'):
                                new_cont = cont[:8] + f"{t:8.4f}" + f"{t:8.4f}" + (cont[24:] if len(cont) > 24 else '\n')
                                new_lines.append(new_cont)
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
                        i += 1
                        continue
                except:
                    pass

            new_lines.append(line)
            i += 1

        with open(output_bdf, 'w', encoding='latin-1') as f:
            f.writelines(new_lines)

        return output_bdf

    def _apply_offsets(self, bdf_path, folder):
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
                t = self.current_skin_thicknesses.get(pid, float(self.skin_min_thickness.get()))
                landing_offsets[eid] = -t / 2.0

                if elem.type in ['CQUAD4', 'CTRIA3']:
                    try:
                        nids = elem.node_ids[:3]
                        nodes = [bdf.nodes[n] for n in nids if n in bdf.nodes]
                        if len(nodes) >= 3:
                            p1, p2, p3 = [np.array(n.get_position()) for n in nodes]
                            normal = np.cross(p2-p1, p3-p1)
                            nl = np.linalg.norm(normal)
                            if nl > 1e-10:
                                landing_normals[eid] = normal / nl
                    except:
                        pass

            # Bar offsets
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
                bar_t = self.current_bar_thicknesses.get(pid, float(self.bar_min_thickness.get()))

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
                        bar_offsets[eid] = tuple(-best_normal * offset_mag)

            # Apply to BDF
            with open(bdf_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            def fmt(v, w=8):
                s = f"{v:.4f}"
                return s[:w].ljust(w) if len(s) <= w else f"{v:.2E}"[:w].ljust(w)

            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]

                if line.startswith('CQUAD4'):
                    try:
                        eid = int(line[8:16].strip())
                        if eid in landing_offsets:
                            new_line = line[:64] + fmt(landing_offsets[eid]) + (line[72:] if len(line) > 72 else '\n')
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

            output_bdf = os.path.join(folder, "model_offset.bdf")
            with open(output_bdf, 'w', encoding='latin-1') as f:
                f.writelines(new_lines)

            return output_bdf

        except Exception as e:
            self.log(f"    Offset error: {e}")
            return bdf_path

    def _run_nastran(self, bdf_path, folder):
        nastran = self.nastran_path.get()
        if not nastran or not os.path.exists(nastran):
            return False
        try:
            cmd = f'"{nastran}" "{bdf_path}" out="{folder}" scratch=yes batch=no'
            proc = subprocess.Popen(cmd, shell=True)
            proc.wait(timeout=600)
            return True
        except:
            return False

    def _extract_stresses(self, folder):
        results = []
        for f in os.listdir(folder):
            if f.lower().endswith('.op2'):
                try:
                    op2 = OP2(debug=False)
                    op2.read_op2(os.path.join(folder, f))

                    if hasattr(op2, 'cbar_stress') and op2.cbar_stress:
                        for sc_id, data in op2.cbar_stress.items():
                            for i, eid in enumerate(data.element):
                                stress = data.data[0, i, 0] if len(data.data.shape) == 3 else data.data[i, 0]
                                results.append({'eid': int(eid), 'type': 'bar', 'stress': float(stress)})

                    if hasattr(op2, 'cquad4_stress') and op2.cquad4_stress:
                        for sc_id, data in op2.cquad4_stress.items():
                            for i, eid in enumerate(data.element):
                                stress = data.data[0, i, -1] if len(data.data.shape) == 3 else data.data[i, -1]
                                results.append({'eid': int(eid), 'type': 'shell', 'stress': float(stress)})
                except:
                    pass
        return results

    def _calculate_rf(self, stresses, target_rf):
        details = []
        failing_pids = set()
        elem_fit = prop_fit = 0

        for s in stresses:
            eid = s['eid']
            stress = abs(s['stress'])
            etype = s['type']
            pid = self.elem_to_prop.get(eid)

            if etype == 'bar' and pid:
                t = self.current_bar_thicknesses.get(pid, float(self.bar_min_thickness.get()))
            else:
                t = self.current_skin_thicknesses.get(pid, float(self.skin_min_thickness.get())) if pid else float(self.skin_min_thickness.get())

            # Try element fit first, then property fit
            allowable = None
            fit_src = "none"

            if eid in self.allowable_elem_interp:
                allowable = self.get_allowable_stress_elem(eid, t)
                if allowable:
                    fit_src = "element"
                    elem_fit += 1

            if allowable is None and pid:
                allowable = self.get_allowable_stress(pid, t)
                if allowable:
                    fit_src = "property"
                    prop_fit += 1

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

            req_t = self.get_required_thickness(pid, stress, target_rf) if pid and status == 'FAIL' else None

            details.append({
                'eid': eid, 'pid': pid, 'type': etype, 'thickness': t,
                'stress': stress, 'allowable': allowable, 'rf': rf,
                'status': status, 'fit_src': fit_src, 'req_thickness': req_t
            })

        valid_rf = [d['rf'] for d in details if 0 < d['rf'] < 999]
        min_rf = min(valid_rf) if valid_rf else 0
        n_fail = sum(1 for d in details if d['status'] == 'FAIL')

        return {'min_rf': min_rf, 'n_fail': n_fail, 'details': details, 'failing_pids': failing_pids}

    def _calculate_weight(self):
        weight = 0.0

        for pid in self.skin_properties:
            t = self.current_skin_thicknesses.get(pid, 0)
            rho = self.get_density(pid)
            if pid in self.prop_elements:
                area = sum(self.element_areas.get(eid, 0) for eid in self.prop_elements[pid])
                weight += area * t * rho

        for pid in self.bar_properties:
            t = self.current_bar_thicknesses.get(pid, 0)
            rho = self.get_density(pid)
            if pid in self.prop_elements:
                length = sum(self.bar_lengths.get(eid, 0) for eid in self.prop_elements[pid])
                weight += length * t * t * rho

        return weight

    def _smart_thickness_update(self, result, step, bar_min, bar_max, skin_min, skin_max, target_rf, rf_tol):
        """Per-property smart thickness update based on RF sensitivity."""
        details = result.get('rf_details', [])
        failing_pids = result.get('failing_pids', set())

        # Group by PID and find min RF per property
        pid_data = {}
        for d in details:
            pid = d['pid']
            if pid is None:
                continue
            if pid not in pid_data:
                pid_data[pid] = {'min_rf': 999, 'max_stress': 0, 'req_thickness': None}
            if d['rf'] < pid_data[pid]['min_rf']:
                pid_data[pid]['min_rf'] = d['rf']
                pid_data[pid]['max_stress'] = d['stress']
                pid_data[pid]['req_thickness'] = d.get('req_thickness')

        updated = 0

        # Increase failing properties
        for pid in failing_pids:
            if pid in self.current_bar_thicknesses:
                current = self.current_bar_thicknesses[pid]
                req_t = pid_data.get(pid, {}).get('req_thickness')
                if req_t and req_t > current:
                    new_t = min(req_t * 1.05, bar_max)  # 5% margin
                else:
                    new_t = min(current + step, bar_max)
                if new_t > current:
                    self.current_bar_thicknesses[pid] = new_t
                    updated += 1

            elif pid in self.current_skin_thicknesses:
                current = self.current_skin_thicknesses[pid]
                req_t = pid_data.get(pid, {}).get('req_thickness')
                if req_t and req_t > current:
                    new_t = min(req_t * 1.05, skin_max)
                else:
                    new_t = min(current + step, skin_max)
                if new_t > current:
                    self.current_skin_thicknesses[pid] = new_t
                    updated += 1

        # Decrease over-designed properties (RF > target + tolerance + 0.2)
        reduce_threshold = target_rf + rf_tol + 0.2
        for pid, data in pid_data.items():
            if data['min_rf'] > reduce_threshold and pid not in failing_pids:
                if pid in self.current_bar_thicknesses:
                    current = self.current_bar_thicknesses[pid]
                    new_t = max(current - step/2, bar_min)
                    if new_t < current:
                        self.current_bar_thicknesses[pid] = new_t
                        updated += 1

                elif pid in self.current_skin_thicknesses:
                    current = self.current_skin_thicknesses[pid]
                    new_t = max(current - step/2, skin_min)
                    if new_t < current:
                        self.current_skin_thicknesses[pid] = new_t
                        updated += 1

        if updated:
            self.log(f"  Updated {updated} properties")

    def _save_iteration(self, folder, result):
        try:
            with open(os.path.join(folder, "summary.csv"), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Parameter', 'Value'])
                w.writerow(['Iteration', result['iteration']])
                w.writerow(['Min_RF', result['min_rf']])
                w.writerow(['N_Fail', result['n_fail']])
                w.writerow(['Weight', result['weight']])

            pd.DataFrame(result['rf_details']).to_csv(os.path.join(folder, "rf_details.csv"), index=False)

            bar_data = [{'PID': p, 'Thickness': t} for p, t in result['bar_thicknesses'].items()]
            skin_data = [{'PID': p, 'Thickness': t} for p, t in result['skin_thicknesses'].items()]
            pd.DataFrame(bar_data).to_csv(os.path.join(folder, "bar_thicknesses.csv"), index=False)
            pd.DataFrame(skin_data).to_csv(os.path.join(folder, "skin_thicknesses.csv"), index=False)

        except Exception as e:
            self.log(f"  Save error: {e}")

    def _save_results(self, folder):
        try:
            history = [{'iteration': r['iteration'], 'min_rf': r['min_rf'], 'n_fail': r['n_fail'], 'weight': r['weight']} for r in self.iteration_results]
            pd.DataFrame(history).to_csv(os.path.join(folder, "history.csv"), index=False)

            if self.best_solution:
                pd.DataFrame([{'iteration': self.best_solution['iteration'], 'min_rf': self.best_solution['min_rf'], 'weight': self.best_solution['weight']}]).to_csv(os.path.join(folder, "best.csv"), index=False)

        except Exception as e:
            self.log(f"Save error: {e}")

    def _update_ui(self):
        if self.best_solution:
            self.result_summary.config(text=f"Best: Weight={self.best_solution['weight']:.6f}t, RF={self.best_solution['min_rf']:.4f}", foreground="green")

            self.best_text.delete(1.0, tk.END)
            txt = f"Best Solution (Iteration {self.best_solution['iteration']}):\n"
            txt += f"  Weight: {self.best_solution['weight']:.6f} tonnes\n"
            txt += f"  Min RF: {self.best_solution['min_rf']:.4f}\n"
            txt += f"  Failures: {self.best_solution['n_fail']}\n\n"

            txt += "Bar Thicknesses (sample):\n"
            for pid in list(self.best_solution['bar_thicknesses'].keys())[:10]:
                txt += f"  PID {pid}: {self.best_solution['bar_thicknesses'][pid]:.2f} mm\n"

            txt += "\nSkin Thicknesses (sample):\n"
            for pid in list(self.best_solution['skin_thicknesses'].keys())[:10]:
                txt += f"  PID {pid}: {self.best_solution['skin_thicknesses'][pid]:.2f} mm\n"

            self.best_text.insert(tk.END, txt)

    def export_results(self):
        if not self.iteration_results:
            messagebox.showerror("Error", "No results")
            return

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_folder.get(), f"results_{ts}.xlsx")

            with pd.ExcelWriter(path, engine='openpyxl') as w:
                history = [{'iteration': r['iteration'], 'min_rf': r['min_rf'], 'n_fail': r['n_fail'], 'weight': r['weight']} for r in self.iteration_results]
                pd.DataFrame(history).to_excel(w, sheet_name='History', index=False)

                if self.best_solution:
                    bar_data = [{'PID': p, 'Thickness': t} for p, t in self.best_solution['bar_thicknesses'].items()]
                    skin_data = [{'PID': p, 'Thickness': t} for p, t in self.best_solution['skin_thicknesses'].items()]
                    pd.DataFrame(bar_data).to_excel(w, sheet_name='Bar_Thicknesses', index=False)
                    pd.DataFrame(skin_data).to_excel(w, sheet_name='Skin_Thicknesses', index=False)

            self.log(f"\nExported: {path}")
            messagebox.showinfo("Export", f"Saved to:\n{path}")

        except Exception as e:
            self.log(f"Export error: {e}")


def main():
    root = tk.Tk()
    app = ThicknessIterationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
