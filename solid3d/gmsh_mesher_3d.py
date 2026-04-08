# solid3d/gmsh_mesher_3d.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.plate import Plate
from solid3d.geometry_abh_3d import PlateABH3DGeometry
from solid3d.mesh_data_3d import Mesh3DData


@dataclass
class Solid3DMeshOptions:
    """
    Paramètres du maillage 3D volumique.
    Toutes les longueurs sont en mètres.
    """

    element_order: int = 2
    global_size: float = 0.020
    local_size: float = 0.005
    local_refinement_radius: float = 0.080
    transition_thickness: float = 0.030

    top_surface_nu: int = 21
    top_surface_nv: int = 17

    algorithm_3d: int = 10
    save_msh_path: str | None = None
    reuse_saved_msh: bool = True

    # --- options high-order ---
    optimize_high_order: bool = True
    high_order_opt_mode: int = 2          # 1=optimization, 2=elastic+optimization, 3=elastic, 4=fast curving
    high_order_num_layers: int = 6
    high_order_pass_max: int = 25
    high_order_threshold_min: float = 0.1
    high_order_threshold_max: float = 2.0
    high_order_fix_boundary_nodes: int = 0
    high_order_prim_surf_mesh: int = 1
    high_order_iter_max: int = 100

    def __post_init__(self) -> None:
        if self.element_order not in (1, 2):
            raise ValueError("element_order doit valoir 1 ou 2.")
        if self.global_size <= 0.0:
            raise ValueError("global_size doit être > 0.")
        if self.local_size <= 0.0:
            raise ValueError("local_size doit être > 0.")
        if self.local_size > self.global_size:
            raise ValueError("local_size doit être <= global_size.")
        if self.local_refinement_radius < 0.0:
            raise ValueError("local_refinement_radius doit être >= 0.")
        if self.transition_thickness < 0.0:
            raise ValueError("transition_thickness doit être >= 0.")
        if self.top_surface_nu < 4 or self.top_surface_nv < 4:
            raise ValueError("top_surface_nu et top_surface_nv doivent être >= 4.")
        if self.high_order_opt_mode not in (0, 1, 2, 3, 4):
            raise ValueError("high_order_opt_mode doit être dans {0,1,2,3,4}.")


class GmshABH3DMesher:
    """
    Générateur Gmsh pour la plaque 3D avec VBH usiné par le dessus.

    Remarque importante :
    la face supérieure est construite ici comme une surface BSpline OCC,
    tronquée par le contour rectangulaire supérieur.
    """
    def _optimize_high_order_mesh(self, options: Solid3DMeshOptions) -> None:
        try:
            import gmsh
        except ImportError as exc:
            raise ImportError("Le module gmsh n'est pas installé. Installe-le avant d'utiliser la branche 3D.") from exc

        if options.element_order <= 1:
            return

        # important : Tetra10 droit, pas courbe
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

        gmsh.model.mesh.setOrder(options.element_order)

        # pour le premier debug : ne pas relancer l'optimisation high-order
        if not options.optimize_high_order:
            return

        gmsh.option.setNumber("Mesh.HighOrderOptimize", options.high_order_opt_mode)
        gmsh.option.setNumber("Mesh.HighOrderNumLayers", options.high_order_num_layers)
        gmsh.option.setNumber("Mesh.HighOrderPassMax", options.high_order_pass_max)
        gmsh.option.setNumber("Mesh.HighOrderThresholdMin", options.high_order_threshold_min)
        gmsh.option.setNumber("Mesh.HighOrderThresholdMax", options.high_order_threshold_max)
        gmsh.option.setNumber("Mesh.HighOrderFixBoundaryNodes", options.high_order_fix_boundary_nodes)
        gmsh.option.setNumber("Mesh.HighOrderPrimSurfMesh", options.high_order_prim_surf_mesh)
        gmsh.option.setNumber("Mesh.HighOrderIterMax", options.high_order_iter_max)

        gmsh.model.mesh.optimize("HighOrderElastic", force=True, niter=1)
        gmsh.model.mesh.optimize("HighOrder", force=True, niter=1)
        
    def __init__(
        self,
        plate: Plate,
        use_black_hole: bool = True,
        model_name: str = "plate_abh_3d",
    ) -> None:
        self.plate = plate
        self.use_black_hole = use_black_hole
        self.model_name = model_name
        self.geometry = PlateABH3DGeometry(plate=plate, use_black_hole=use_black_hole)

    # ------------------------------------------------------------------
    # Helpers géométriques
    # ------------------------------------------------------------------

    @staticmethod
    def _add_planar_quad_surface(occ, corners: list[tuple[float, float, float]]) -> int:
        """
        Crée une surface plane quadrilatère à partir de 4 coins.
        Les points sont donnés dans l'ordre du contour.
        """
        pts = [occ.addPoint(x, y, z) for x, y, z in corners]
        lines = [
            occ.addLine(pts[0], pts[1]),
            occ.addLine(pts[1], pts[2]),
            occ.addLine(pts[2], pts[3]),
            occ.addLine(pts[3], pts[0]),
        ]
        loop = occ.addCurveLoop(lines)
        return occ.addPlaneSurface([loop])

    def _build_occ_volume(self, options: Solid3DMeshOptions) -> dict[str, int]:
        """
        Construit la géométrie OCC :
        - fond plan
        - dessus BSpline usiné
        - 4 faces latérales planes
        - volume fermé
        """
        try:
            import gmsh
        except ImportError as exc:
            raise ImportError("Le module gmsh n'est pas installé. Installe-le avant d'utiliser la branche 3D.") from exc

        occ = gmsh.model.occ

        Lx = self.geometry.Lx
        Ly = self.geometry.Ly
        h0 = self.geometry.h0

        # -------------------------
        # Face inférieure plane
        # -------------------------
        b00 = occ.addPoint(0.0, 0.0, 0.0)
        b10 = occ.addPoint(Lx, 0.0, 0.0)
        b11 = occ.addPoint(Lx, Ly, 0.0)
        b01 = occ.addPoint(0.0, Ly, 0.0)

        bottom_lines = [
            occ.addLine(b00, b10),
            occ.addLine(b10, b11),
            occ.addLine(b11, b01),
            occ.addLine(b01, b00),
        ]
        bottom_loop = occ.addCurveLoop(bottom_lines)
        bottom_surface = occ.addPlaneSurface([bottom_loop])

        # -------------------------
        # Contour supérieur rectangulaire (à z = h0 sur le bord)
        # -------------------------
        t00 = occ.addPoint(0.0, 0.0, h0)
        t10 = occ.addPoint(Lx, 0.0, h0)
        t11 = occ.addPoint(Lx, Ly, h0)
        t01 = occ.addPoint(0.0, Ly, h0)

        top_wire_lines = [
            occ.addLine(t00, t10),
            occ.addLine(t10, t11),
            occ.addLine(t11, t01),
            occ.addLine(t01, t00),
        ]
        top_wire = occ.addWire(top_wire_lines, checkClosed=True)

        # -------------------------
        # Surface supérieure BSpline
        # -------------------------
        X, Y, Z = self.geometry.make_top_grid(
            nu=options.top_surface_nu,
            nv=options.top_surface_nv,
        )

        point_tags = []
        for j in range(options.top_surface_nv):
            for i in range(options.top_surface_nu):
                x = float(X[j, i])
                y = float(Y[j, i])
                z = float(Z[j, i])

                # réutilisation des coins du contour supérieur
                if i == 0 and j == 0:
                    point_tags.append(t00)
                elif i == options.top_surface_nu - 1 and j == 0:
                    point_tags.append(t10)
                elif i == options.top_surface_nu - 1 and j == options.top_surface_nv - 1:
                    point_tags.append(t11)
                elif i == 0 and j == options.top_surface_nv - 1:
                    point_tags.append(t01)
                else:
                    point_tags.append(occ.addPoint(x, y, z))

        top_surface = occ.addBSplineSurface(
            point_tags,
            options.top_surface_nu,
            degreeU=3,
            degreeV=3,
            wireTags=[top_wire],
            wire3D=True,
        )

        # -------------------------
        # Faces latérales planes
        # -------------------------
        front_surface = self._add_planar_quad_surface(
            occ,
            [
                (0.0, 0.0, 0.0),
                (Lx, 0.0, 0.0),
                (Lx, 0.0, h0),
                (0.0, 0.0, h0),
            ],
        )

        right_surface = self._add_planar_quad_surface(
            occ,
            [
                (Lx, 0.0, 0.0),
                (Lx, Ly, 0.0),
                (Lx, Ly, h0),
                (Lx, 0.0, h0),
            ],
        )

        back_surface = self._add_planar_quad_surface(
            occ,
            [
                (Lx, Ly, 0.0),
                (0.0, Ly, 0.0),
                (0.0, Ly, h0),
                (Lx, Ly, h0),
            ],
        )

        left_surface = self._add_planar_quad_surface(
            occ,
            [
                (0.0, Ly, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, h0),
                (0.0, Ly, h0),
            ],
        )

        # Le shell est cousu ("sewing=True") pour tolérer des frontières
        # géométriquement identiques mais topologiquement distinctes.
        shell = occ.addSurfaceLoop(
            [
                bottom_surface,
                top_surface,
                front_surface,
                right_surface,
                back_surface,
                left_surface,
            ],
            sewing=True,
        )

        volume = occ.addVolume([shell])

        occ.synchronize()

        return {
            "bottom_surface": bottom_surface,
            "top_surface": top_surface,
            "front_surface": front_surface,
            "right_surface": right_surface,
            "back_surface": back_surface,
            "left_surface": left_surface,
            "volume": volume,
        }

    def _apply_mesh_controls(self, options: Solid3DMeshOptions) -> None:
        """
        Définit la stratégie de taille de maille.
        """
        try:
            import gmsh
        except ImportError as exc:
            raise ImportError("Le module gmsh n'est pas installé. Installe-le avant d'utiliser la branche 3D.") from exc

        gmsh.option.setNumber("General.Terminal", 1)

        # Important pour éviter qu'une petite taille imposée sur une frontière
        # se propage partout dans le volume.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        gmsh.option.setNumber("Mesh.MeshSizeMin", min(options.local_size, options.global_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(options.local_size, options.global_size))
        gmsh.option.setNumber("Mesh.Algorithm3D", options.algorithm_3d)

        bh = self.plate.black_hole if self.use_black_hole else None

        if bh is not None:
            field = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(field, "XCenter", bh.xc)
            gmsh.model.mesh.field.setNumber(field, "YCenter", bh.yc)
            gmsh.model.mesh.field.setNumber(field, "ZCenter", 0.5 * self.plate.h0)

            # zone raffinée = trou noir + marge de raffinement
            gmsh.model.mesh.field.setNumber(
                field,
                "Radius",
                bh.radius + options.local_refinement_radius,
            )
            gmsh.model.mesh.field.setNumber(field, "Thickness", options.transition_thickness)
            gmsh.model.mesh.field.setNumber(field, "VIn", options.local_size)
            gmsh.model.mesh.field.setNumber(field, "VOut", options.global_size)
            gmsh.model.mesh.field.setAsBackgroundMesh(field)
        else:
            field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(field, "F", f"{options.global_size}")
            gmsh.model.mesh.field.setAsBackgroundMesh(field)

    def _extract_best_tetra_block(self, requested_order: int) -> Mesh3DData:
        """
        Extrait tous les blocs tétraédriques volumiques pertinents
        et compacte les nœuds réellement utilisés.
        """
        try:
            import gmsh
        except ImportError as exc:
            raise ImportError("Le module gmsh n'est pas installé. Installe-le avant d'utiliser la branche 3D.") from exc
        import numpy as np

        # Tous les nœuds du maillage Gmsh
        node_tags_all, node_coords_all, _ = gmsh.model.mesh.getNodes()
        points_all = np.asarray(node_coords_all, dtype=float).reshape(-1, 3)
        node_tags_all = np.asarray(node_tags_all, dtype=np.int64)

        tag_to_global = {int(tag): i for i, tag in enumerate(node_tags_all.tolist())}

        # Tous les éléments volumiques
        element_types, element_tags_blocks, element_node_tags_blocks = gmsh.model.mesh.getElements(dim=3)

        candidate_blocks = []

        for etype, etags, enodes in zip(element_types, element_tags_blocks, element_node_tags_blocks):
            name, dim, order, num_nodes, *_ = gmsh.model.mesh.getElementProperties(etype)

            if dim != 3:
                continue
            if "tetra" not in name.lower():
                continue

            etags = np.asarray(etags, dtype=np.int64)
            enodes = np.asarray(enodes, dtype=np.int64).reshape(-1, num_nodes)

            candidate_blocks.append(
                {
                    "gmsh_element_type": int(etype),
                    "name": name,
                    "order": int(order),
                    "num_nodes": int(num_nodes),
                    "element_tags": etags,
                    "element_node_tags": enodes,
                }
            )

        if not candidate_blocks:
            raise RuntimeError("Aucun bloc tétraédrique 3D n'a été trouvé dans le maillage Gmsh.")

        # On garde TOUS les blocs de l'ordre demandé ;
        # sinon tous les blocs de l'ordre max disponible
        exact = [b for b in candidate_blocks if b["order"] == requested_order]
        if exact:
            selected = exact
        else:
            max_order = max(b["order"] for b in candidate_blocks)
            selected = [b for b in candidate_blocks if b["order"] == max_order]

        # Si plusieurs types existent, on garde celui avec le plus de nœuds/élément
        max_num_nodes = max(b["num_nodes"] for b in selected)
        selected = [b for b in selected if b["num_nodes"] == max_num_nodes]

        num_nodes = int(selected[0]["num_nodes"])
        order = int(selected[0]["order"])
        gmsh_element_type = int(selected[0]["gmsh_element_type"])
        name = selected[0]["name"]

        element_tags = np.concatenate([b["element_tags"] for b in selected])
        element_node_tags = np.vstack([b["element_node_tags"] for b in selected])

        # Remappage Gmsh Tetra10 -> ordre interne attendu
        # interne : [1,2,3,4,12,23,31,14,24,34]
        if num_nodes == 10:
            element_node_tags = element_node_tags[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]

        # Compactage : on ne garde que les nœuds réellement utilisés
        used_node_tags = np.unique(element_node_tags.ravel())
        used_global_idx = np.array([tag_to_global[int(tag)] for tag in used_node_tags], dtype=np.int64)
        points = points_all[used_global_idx]

        tag_to_local = {int(tag): i for i, tag in enumerate(used_node_tags.tolist())}
        conn = np.vectorize(tag_to_local.__getitem__, otypes=[np.int64])(element_node_tags)

        return Mesh3DData(
            points=points,
            cells=conn,
            n_nodes_per_cell=num_nodes,
            gmsh_element_type=gmsh_element_type,
            element_order=order,
            node_tags_gmsh=used_node_tags,
            element_tags_gmsh=element_tags,
            metadata={
                "gmsh_element_name": name,
                "requested_order": requested_order,
                "n_blocks_merged": len(selected),
            },
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def generate(self, options: Solid3DMeshOptions) -> Mesh3DData:
        try:
            import gmsh
        except ImportError as exc:
            raise ImportError("Le module gmsh n'est pas installé. Installe-le avant d'utiliser la branche 3D.") from exc

        msh_path = Path(options.save_msh_path) if options.save_msh_path else None

        gmsh.initialize()
        try:
            if (
                msh_path is not None
                and options.reuse_saved_msh
                and msh_path.exists()
            ):
                gmsh.open(str(msh_path))
                return self._extract_best_tetra_block(requested_order=options.element_order)

            gmsh.model.add(self.model_name)

            self._build_occ_volume(options)
            self._apply_mesh_controls(options)

            gmsh.model.mesh.generate(3)
            self._optimize_high_order_mesh(options)

            if msh_path is not None:
                msh_path.parent.mkdir(parents=True, exist_ok=True)
                gmsh.write(str(msh_path))

            return self._extract_best_tetra_block(requested_order=options.element_order)

        finally:
            gmsh.finalize()