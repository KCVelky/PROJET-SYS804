# fem/__init__.py

from .geometry import PlateGeometry
from .mesh_generator import MeshData, StructuredTriMeshGenerator
from .element_matrices import MindlinTri3Element
from .assembler import GlobalAssembler
from .boundary_conditions import (
    find_boundary_nodes,
    constrained_dofs_from_bc,
    free_dofs_from_constrained,
)
from .fem_model import FEMModel