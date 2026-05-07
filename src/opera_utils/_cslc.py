from __future__ import annotations

import functools
import json
import logging
import re
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from os import fspath
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from pyproj import CRS, Transformer
from shapely import geometry, ops, wkt

try:
    from isce3.core import DateTime, Orbit, StateVector

    HAS_ICE3 = True
except ImportError:
    HAS_ICE3 = False

try:
    from osgeo import gdal, osr

    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    gdal = None
    osr = None

from ._types import Filename
from ._utils import format_nc_filename
from .bursts import normalize_burst_id
from .constants import (
    COMPASS_FILE_REGEX,
    COMPRESSED_CSLC_S1_FILE_REGEX,
    CSLC_S1_FILE_REGEX,
    NISAR_BOUNDING_POLYGON,
    NISAR_FILE_REGEX,
    NISAR_SDS_FILE_REGEX,
    OPERA_IDENTIFICATION,
)

__all__ = [
    "CslcParseError",
    "create_nodata_mask",
    "get_cslc_orbit",
    "get_cslc_polygon",
    "get_lonlat_grid",
    "get_orbit_arrays",
    "get_radar_wavelength",
    "get_union_polygon",
    "get_xy_coords",
    "get_zero_doppler_time",
    "make_nodata_mask",  # TODO: deprecate
    "parse_filename",
]
logger = logging.getLogger(__name__)


class CslcParseError(ValueError):
    """Error raised for non-matching filename."""


def parse_filename(h5_filename: Filename) -> dict[str, str | datetime]:
    """Parse the filename of a CSLC HDF5 file.

    Parameters
    ----------
    h5_filename : Filename
        The path or name of the CSLC HDF5 file.

    Returns
    -------
    dict[str, str | datetime]
        A dictionary containing parsed components of the filename:
        - project: str
        - level: str
        - product_type: str
        - burst_id: str (normalized to lowercase with underscores)
        - start_datetime: datetime
        - generation_datetime: datetime
        - sensor: str
        - polarization: str
        - product_version: str

    Or, if the filename is a COMPASS-generated file,
        - burst_id: str (lowercase with underscores)
        - start_datetime: datetime (but no hour/minute/second info)

    Raises
    ------
    CslcParseError
        If the filename does not match the expected pattern.

    """
    name = Path(h5_filename).name
    match: re.Match | None = None

    if (match := re.match(CSLC_S1_FILE_REGEX, name)) or (
        match := re.match(COMPRESSED_CSLC_S1_FILE_REGEX, name)
    ):
        return _parse_cslc_product(match)
    elif match := re.match(NISAR_FILE_REGEX, name):
        return _parse_gslc_product(match)
    elif match := re.match(NISAR_SDS_FILE_REGEX, name):
        return _parse_nisar_sds_product(match)
    elif match := re.match(COMPASS_FILE_REGEX, name):
        return _parse_compass(match)
    else:
        msg = f"Unable to parse {h5_filename}"
        raise CslcParseError(msg)


def _parse_compass(match: re.Match):
    result = match.groupdict()
    result["start_datetime"] = datetime.strptime(
        result["start_datetime"], "%Y%m%d"
    ).replace(tzinfo=timezone.utc)
    return result


def _parse_gslc_product(match: re.Match):
    result = match.groupdict()
    result["frame_id"] = result["frame_id"]
    fmt = "%Y%m%dT%H%M%SZ"
    result["start_datetime"] = datetime.strptime(result["start_datetime"], fmt).replace(
        tzinfo=timezone.utc
    )
    result["generation_datetime"] = datetime.strptime(
        result["generation_datetime"], fmt
    ).replace(tzinfo=timezone.utc)
    return result


def _parse_nisar_sds_product(match: re.Match):
    """Parse official NISAR SDS product filename format."""
    result = match.groupdict()
    # Set sensor to "NI" for NISAR products (used by product.py for input_sensors)
    result["sensor"] = "NI"
    # Construct frame_id from relative_orbit and frame
    result["frame_id"] = f"{result['relative_orbit']}_{result['frame']}"
    # Parse datetime without Z suffix
    fmt = "%Y%m%dT%H%M%S"
    result["start_datetime"] = datetime.strptime(result["start_datetime"], fmt).replace(
        tzinfo=timezone.utc
    )
    result["end_datetime"] = datetime.strptime(result["end_datetime"], fmt).replace(
        tzinfo=timezone.utc
    )
    return result


def _parse_cslc_product(match: re.Match):
    result = match.groupdict()
    # Normalize to lowercase / underscore
    result["burst_id"] = normalize_burst_id(result["burst_id"])
    fmt = "%Y%m%dT%H%M%SZ"
    result["start_datetime"] = datetime.strptime(result["start_datetime"], fmt).replace(
        tzinfo=timezone.utc
    )
    result["generation_datetime"] = datetime.strptime(
        result["generation_datetime"], fmt
    ).replace(tzinfo=timezone.utc)
    return result


def get_dataset_name(h5_filename: Filename) -> str:
    """Get the complex valued dataset from the CSLC HDF5 file.

    Parameters
    ----------
    h5_filename : Filename
        The path or name of the CSLC HDF5 file.

    Returns
    -------
    str
        The name of the complex dataset in the format "/data/{polarization}".

    Raises
    ------
    CslcParseError
        If the filename cannot be parsed.

    """
    name = Path(h5_filename).name
    parsed = parse_filename(name)
    if "polarization" in parsed:
        return f"/data/{parsed['polarization']}"
    else:
        # For compass, no polarization is given, so we have to check the file
        with h5py.File(h5_filename) as hf:
            if "VV" in hf["/data"]:
                return "/data/VV"
            else:
                return "/data/HH"


def get_zero_doppler_time(
    filename: Filename,
    dataset: str | None = None,
    type_: str = "start",
    datetime_format: str = "%Y-%m-%d %H:%M:%S.%f",
) -> datetime:
    """Get the full acquisition time from the CSLC product.

    Uses `/identification/zero_doppler_{type_}_time` from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    dataset : str, optional
        The subdataset to read zero doppler time from
    type_ : str, optional
        Either "start" or "stop", by default "start".
    datetime_format : str, optional
        The format of the datetime

    Returns
    -------
    str
        Full acquisition time.

    """

    def get_dt(in_str):
        # Sentinel-1: datetime_format = "%Y-%m-%d %H:%M:%S.%f"
        # NISAR: datetime_format = "%Y-%m-%dT%H:%M:%S.%f" + some extra digits
        # The index 0:26 gets rid of those extra digits
        return datetime.strptime(in_str.decode("utf-8")[:26], datetime_format)

    # This has to be the default but since we have to change in disp-s1
    # product.py and maybe other use cases, I am leaving it as is
    # because it depends on the type_
    dset = f"/identification/zero_doppler_{type_}_time"
    if dataset:
        dset = dataset
    value = _get_dset_and_attrs(filename, dset, parse_func=get_dt)[0]
    return value


def _get_dset_and_attrs(
    filename: Filename,
    dset_name: str,
    parse_func: Callable = lambda x: x,
) -> tuple[Any, dict[str, Any]]:
    """Get one dataset's value and attributes from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    dset_name : str
        Name of the dataset.
    parse_func : Callable, optional
        Function to parse the dataset value, by default lambda x: x
        For example, could be parse_func=lambda x: x.decode("utf-8") to decode,
        or getting a datetime object from a string.

    Returns
    -------
    dset : Any
        The value of the scalar
    attrs : dict
        Attributes.

    """
    # Handle VSI paths with GDAL's multidim API (h5py can't open /vsis3).
    filename_str = str(filename)
    if filename_str.startswith("/vsi"):
        if not HAS_GDAL:
            msg = "GDAL is required to read VSI paths but is not installed"
            raise ImportError(msg)

        raw_value, attrs = _read_scalar_mdarray(filename_str, dset_name)
        if raw_value is None:
            msg = f"Could not open {dset_name} from {filename_str}"
            raise ValueError(msg)
        value = parse_func(raw_value)
        return value, attrs
    else:
        # Use h5py for local files
        with h5py.File(filename, "r") as hf:
            dset = hf[dset_name]
            attrs = dict(dset.attrs)
            value = parse_func(dset[()])
            return value, attrs


def get_radar_wavelength(filename: Filename) -> float:
    """Get the radar wavelength from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.

    Returns
    -------
    wavelength : float
        Radar wavelength in meters.

    """
    dset = "/metadata/processing_information/input_burst_metadata/wavelength"
    value = _get_dset_and_attrs(filename, dset)[0]
    return value


def get_orbit_arrays(
    h5file: Filename,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    """Retrieve orbit arrays and timestamps from an HDF5 file.

    This function parses the filename to determine the sensor type
    (e.g., "S1" or "NISAR") and calls either `get_nisar_orbit` or
    `get_s1_orbit` accordingly.

    Parameters
    ----------
    h5file : Filename
        Path to the HDF5 file containing orbit data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, datetime]
        - Position array (np.ndarray)
        - Velocity array (np.ndarray)
        - Time array (np.ndarray)
        - Reference datetime (datetime)

    """
    # Parse the filename to figure out if this is S1 vs NISAR
    parsed = parse_filename(h5file)
    project = str(parsed.get("project", ""))
    if project.lower().startswith("nisar"):
        return get_nisar_orbit(h5file)
    else:
        return get_s1_orbit(h5file)


def get_s1_orbit(
    h5file: Filename,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    """Parse orbit info from OPERA S1 CSLC HDF5 file into python types.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    times : np.ndarray
        Array of times in seconds since reference epoch.
    positions : np.ndarray
        Array of positions in meters.
    velocities : np.ndarray
        Array of velocities in meters per second.
    reference_epoch : datetime
        Reference epoch of orbit.

    """
    with h5py.File(h5file) as hf:
        orbit_group = hf["/metadata/orbit"]
        times = orbit_group["time"][:]
        positions = np.stack([orbit_group[f"position_{p}"] for p in ["x", "y", "z"]]).T
        velocities = np.stack([orbit_group[f"velocity_{p}"] for p in ["x", "y", "z"]]).T
        reference_epoch = datetime.fromisoformat(
            orbit_group["reference_epoch"][()].decode()
        )
    return times, positions, velocities, reference_epoch


def get_nisar_orbit(
    h5file: Filename,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    """Parse orbit info from NISAR HDF5 file into python types.

    Parameters
    ----------
    h5file : Filename
        Path to NISAR GSLC HDF5 file.

    Returns
    -------
    times : np.ndarray
        Array of times in seconds since reference epoch.
    positions : np.ndarray
        Array of positions in meters.
    velocities : np.ndarray
        Array of velocities in meters per second.
    reference_epoch : datetime
        Reference epoch of orbit.

    """
    with h5py.File(h5file) as hf:
        orbit_group = hf["/science/LSAR/GSLC//metadata/orbit"]
        times = orbit_group["time"][:]
        positions = orbit_group["position"][()]
        velocities = orbit_group["velocity"][()]
        units = orbit_group["time"].attrs["units"].decode("utf-8")
        reference_epoch_str = units.split("since")[-1].strip()
        reference_epoch = datetime.fromisoformat(reference_epoch_str)
    return times, positions, velocities, reference_epoch


def get_cslc_orbit(h5file: Filename):
    """Parse orbit info from OPERA S1 CSLC HDF5 file into an isce3.core.Orbit.

    `isce3` must be installed to use this function.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    orbit : isce3.core.Orbit
        Orbit object.

    """
    if not HAS_ICE3:
        msg = "isce3 must be installed to use this function"
        raise ImportError(msg)

    times, positions, velocities, reference_epoch = get_orbit_arrays(h5file)
    orbit_svs = []

    for t, x, v in zip(times, positions, velocities):
        orbit_svs.append(
            StateVector(
                DateTime(reference_epoch + timedelta(seconds=t)),
                x,
                v,
            )
        )

    return Orbit(orbit_svs)


def get_xy_coords(
    h5file: Filename, subsample: int = 100
) -> tuple[np.ndarray, np.ndarray, int]:
    """Get x and y grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    x : np.ndarray
        Array of x coordinates in meters.
    y : np.ndarray
        Array of y coordinates in meters.
    epsg_code : int
        EPSG code of projection.

    """
    with h5py.File(h5file) as hf:
        x = hf["/data/x_coordinates"][:]
        y = hf["/data/y_coordinates"][:]
        projection_dset = hf["/data/projection"]
        crs_string = ""
        # https://github.com/corteva/rioxarray/blob/5783693895b4b055909c5758a72a5d40a365ef11/rioxarray/rioxarray.py#L34
        for attr_name in "spatial_ref", "crs_wkt":
            if attr_name in projection_dset.attrs:
                crs_string = projection_dset.attrs[attr_name]
        if not crs_string:
            msg = f"Failed to parse CRS for {h5file}"
            raise ValueError(msg)
        if isinstance(crs_string, bytes):
            crs_string = crs_string.decode("utf-8")
        crs = CRS.from_user_input(crs_string)

    return x[::subsample], y[::subsample], crs.to_epsg()


def get_lonlat_grid(
    h5file: Filename, subsample: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Get 2D latitude and longitude grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    lat : np.ndarray
        2D Array of latitude coordinates in degrees.
    lon : np.ndarray
        2D Array of longitude coordinates in degrees.

    """
    x, y, epsg = get_xy_coords(h5file, subsample)
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(xx=xx, yy=yy, radians=False)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat


def get_cslc_polygon(
    opera_file: Filename, buffer_degrees: float = 0.0
) -> geometry.Polygon | None:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0

    """
    if "NISAR" in str(opera_file):
        dset_name = NISAR_BOUNDING_POLYGON
    else:
        dset_name = f"{OPERA_IDENTIFICATION}/bounding_polygon"

    opera_str = str(opera_file)
    if opera_str.startswith("/vsi"):
        # Use GDAL's multidim API for VSI paths — the 2D raster API can't read
        # scalar string MDArrays, and the HDF5 driver does ranged S3 reads.
        if not HAS_GDAL:
            msg = "GDAL is required to read VSI paths but is not installed"
            raise ImportError(msg)

        wkt_str = _read_string_mdarray(opera_str, dset_name)
        if wkt_str is None:
            logger.debug(f"Could not find {dset_name} in {opera_file}")
            return None
    else:
        # Use h5py for local files
        with h5py.File(opera_file) as hf:
            if dset_name not in hf:
                logger.debug(f"Could not find {dset_name} in {opera_file}")
                return None
            wkt_str = hf[dset_name][()].decode("utf-8")

    return wkt.loads(wkt_str).buffer(buffer_degrees)


def get_union_polygon(
    opera_file_list: Sequence[Filename], buffer_degrees: float = 0.0
) -> geometry.Polygon:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0

    """
    # Prefer local files (faster), but VSI paths work too via GDAL's multidim API.
    local_files = [f for f in opera_file_list if not str(f).startswith("/vsi")]

    if local_files:
        files_to_use = local_files
        logger.info(
            f"Using {len(local_files)} local file(s) for nodata mask polygon extraction"
        )
    else:
        files_to_use = list(opera_file_list)
        logger.info(
            f"Using {len(files_to_use)} VSI file(s) for nodata mask polygon extraction"
        )

    # I/O-bound: parallelize across files (each is one /vsis3 ranged GET).
    max_workers = min(16, len(files_to_use)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        polygons = list(
            exc.map(lambda f: get_cslc_polygon(f, buffer_degrees), files_to_use)
        )
    polygons = [p for p in polygons if p is not None]

    if len(polygons) == 0:
        msg = "No polygons found in the given file list."
        raise ValueError(msg)
    # Union all the polygons
    return ops.unary_union(polygons)


def create_nodata_mask(
    opera_file_list: Sequence[Filename],
    out_file: Filename,
    dset_name: str | None = None,
    buffer_pixels: int = 400,
    overwrite: bool = False,
):
    """Create a boolean raster mask from the union of nodata polygons using GDAL.

    The output datatype is UInt8, where 1 means valid data in the polygon and
    0 is invalid (outside the polygon).

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    out_file : Filename
        Output filename.
    dset_name : str, optional
        The name of the dataset in opera files
    buffer_pixels : int, optional
        Number of pixels to buffer the union polygon by, by default 0.
        Note that buffering will *decrease* the numba of pixels marked as nodata.
        This is to be more conservative to not mask possible valid pixels.
    overwrite : bool, optional
        Overwrite the output file if it already exists, by default False

    """
    if not HAS_GDAL:
        msg = "osgeo (GDAL) must be installed to use this function"
        raise ImportError(msg)

    gdal.UseExceptions()
    if Path(out_file).exists():
        if not overwrite:
            logger.debug(f"Skipping {out_file} since it already exists.")
            return
        else:
            logger.info(f"Overwriting {out_file} since overwrite=True.")
            Path(out_file).unlink()

    # Check these are the right format to get nodata polygons
    if dset_name:
        dataset_name = dset_name
    else:
        # For get_dataset_name, need a local file (uses h5py internally)
        # Filter out VSI paths
        local_files = [f for f in opera_file_list if not str(f).startswith("/vsi")]
        reference_file = local_files[0] if local_files else opera_file_list[0]

        logger.info(
            f"Creating nodata mask: found {len(local_files)} local files out of"
            f" {len(opera_file_list)} total"
        )
        logger.info(f"Using reference file for metadata: {reference_file}")

        # Verify the reference file exists if it's a local path
        if (
            not str(reference_file).startswith("/vsi")
            and not Path(reference_file).exists()
        ):
            logger.error(f"Local reference file does not exist: {reference_file}")
            msg = f"Local reference file not found: {reference_file}"
            raise FileNotFoundError(msg)

        try:
            dataset_name = get_dataset_name(reference_file)
        except CslcParseError as e:
            msg = f"{reference_file} is not a CSLC file"
            raise ValueError(msg) from e
        except Exception as e:
            # If h5py fails on a VSI path, provide helpful error message
            if str(reference_file).startswith("/vsi"):
                msg = (
                    f"Cannot open VSI path with h5py: {reference_file}. h5py does not"
                    " support GDAL virtual file systems. Ensure at least one file is"
                    " downloaded locally for metadata extraction."
                )
            else:
                msg = f"Could not get dataset name from {reference_file}: {e}"
            raise ValueError(msg) from e

    # For GDAL operations, can use VSI paths (use last file as before)
    try:
        test_f = format_nc_filename(opera_file_list[-1], dataset_name)
        # convert pixels to degrees lat/lon
        gt = _get_raster_gt(test_f, raw_file=opera_file_list[-1])
    except RuntimeError as e:
        msg = f"Unable to get geotransform from {test_f}"
        raise ValueError(msg) from e
    # TODO: more robust way to get the pixel size... this is a hack
    # maybe just use pyproj to warp lat/lon to meters and back?
    dx_meters = gt[1]
    dx_degrees = dx_meters / 111000
    buffer_degrees = buffer_pixels * dx_degrees

    # Get the union of all the polygons and convert to a temp geojson
    union_poly = get_union_polygon(opera_file_list, buffer_degrees=buffer_degrees)
    # convert shapely polygon to geojson

    # Make a dummy raster from the last file with all 0s
    # This will get filled in with the polygon rasterization
    # Use GDAL Python API directly instead of subprocess (much faster!)
    test_f_str = format_nc_filename(opera_file_list[-1], dataset_name)
    src_ds = gdal.Open(test_f_str, gdal.GA_ReadOnly)
    if src_ds is None:
        msg = f"Could not open {test_f_str} to get dimensions"
        raise ValueError(msg)

    # Get dimensions and georeferencing
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    try:
        projection = src_ds.GetProjection()
    except RuntimeError:
        projection = ""
    try:
        geotransform = src_ds.GetGeoTransform()
    except RuntimeError:
        geotransform = None
    src_ds = None

    # NISAR GSLCs opened via the HDF5 driver don't expose geotransform/projection;
    # fall back to the multidim API to read xCoordinates/yCoordinates/projection.
    if (
        not projection
        or geotransform is None
        or tuple(geotransform) == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ):
        info = _read_nisar_geoinfo_multidim(opera_file_list[-1], dataset_name)
        if info is not None:
            geotransform, projection = info

    if geotransform is None or not projection:
        msg = f"Could not resolve geotransform/projection for {test_f_str}"
        raise ValueError(msg)

    # Create output raster directly with GDAL
    driver = gdal.GetDriverByName("GTiff")
    options = [
        "COMPRESS=LZW",
        "TILED=YES",
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
    ]
    dst_ds = driver.Create(fspath(out_file), xsize, ysize, 1, gdal.GDT_Byte, options)
    if dst_ds is None:
        msg = f"Could not create {out_file}"
        raise ValueError(msg)

    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(projection)

    # CRITICAL: Set SRS for GDAL 3+ compatibility
    # Without this, gdal.Rasterize will fail with SRS mismatch warnings
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    dst_ds.SetSpatialRef(srs)

    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(0)
    # Initialize with zeros (will be overwritten by polygon rasterization)
    dst_band.Fill(0)
    dst_band.FlushCache()
    dst_band = None

    logger.info(f"Created empty mask raster: {out_file}")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vector_file = Path(tmpdir) / "temp.geojson"
        with open(temp_vector_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(geometry.mapping(union_poly)))

        # Open the input vector file
        src_ds = gdal.OpenEx(fspath(temp_vector_file), gdal.OF_VECTOR)

        # Now burn in the union of all polygons (dst_ds is still open)
        gdal.Rasterize(dst_ds, src_ds, burnValues=[1])

    # Close the dataset after rasterization
    dst_ds = None


make_nodata_mask = create_nodata_mask


def _get_raster_gt(filename: Filename, raw_file: Filename | None = None) -> list[float]:
    """Get the geotransform from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load (already formatted for GDAL, e.g. HDF5:"...":...).
    raw_file : Filename, optional
        Unformatted path to the underlying HDF5. Used as a multidim-API fallback
        when the HDF5 driver doesn't synthesize a geotransform (e.g. NISAR GSLCs).

    Returns
    -------
    List[float]
        6 floats representing a GDAL Geotransform.

    """
    if not HAS_GDAL:
        msg = "osgeo (GDAL) must be installed to use this function"
        raise ImportError(msg)

    try:
        ds = gdal.Open(fspath(filename))
        gt = ds.GetGeoTransform()
        needs_fallback = tuple(gt) == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    except RuntimeError:
        gt = None
        needs_fallback = True

    if needs_fallback and raw_file is not None:
        info = _read_nisar_geoinfo_multidim(raw_file)
        if info is not None:
            return list(info[0])
    if gt is None:
        msg = f"Could not read geotransform from {filename}"
        raise RuntimeError(msg)
    return gt


def _read_mdarray_value(ar):
    """Read an MDArray or Attribute; return a typed Python scalar / numpy array.

    `MDArray.Read()` returns a raw bytearray for numeric types, so prefer
    `ReadAsArray()` (which yields a typed numpy array) and fall back to
    `Read()` only for string/variable-length classes it can't expose.
    """
    try:
        arr = ar.ReadAsArray()
    except Exception:
        arr = None
    if arr is not None:
        if isinstance(arr, np.ndarray) and arr.size == 1:
            return arr.item()
        return arr
    val = ar.Read()
    if isinstance(val, (list, tuple)) and len(val) == 1:
        val = val[0]
    if isinstance(val, (bytes, bytearray)):
        try:
            return bytes(val).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(val)
    return val


def _read_scalar_mdarray(
    filename: Filename, dset_path: str
) -> tuple[Any, dict[str, Any]]:
    """Read a scalar MDArray and its attributes via GDAL's multidim API.

    Returns (value, attrs). `value` is a Python scalar (int/float/str/bytes)
    when possible, the raw numpy array otherwise, or None on failure.
    """
    if not HAS_GDAL:
        return None, {}
    parts = [p for p in dset_path.split("/") if p]
    if not parts:
        return None, {}
    ds = grp = ar = None
    try:
        ds = gdal.OpenEx(fspath(filename), gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            return None, {}
        grp = ds.GetRootGroup()
        for name in parts[:-1]:
            grp = grp.OpenGroup(name)
            if grp is None:
                return None, {}
        ar = grp.OpenMDArray(parts[-1])
        if ar is None:
            return None, {}
        val = _read_mdarray_value(ar)
        attrs: dict[str, Any] = {}
        try:
            for a in ar.GetAttributes() or []:
                attrs[a.GetName()] = _read_mdarray_value(a)
        except Exception:
            attrs = {}
    except Exception as e:
        logger.debug(f"_read_scalar_mdarray({filename}, {dset_path}) failed: {e}")
        return None, {}
    finally:
        ar = grp = ds = None

    return val, attrs


def _read_string_mdarray(filename: Filename, dset_path: str) -> str | None:
    """Read a scalar string MDArray from an HDF5 via the multidim API.

    Walks the group hierarchy given by `dset_path` (absolute, e.g.
    `/science/LSAR/identification/boundingPolygon`) and returns the string
    value. Works for /vsis3 paths with ranged reads.
    """
    if not HAS_GDAL:
        return None
    parts = [p for p in dset_path.split("/") if p]
    if len(parts) < 1:
        return None
    ds = grp = ar = None
    try:
        ds = gdal.OpenEx(fspath(filename), gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            return None
        grp = ds.GetRootGroup()
        for name in parts[:-1]:
            grp = grp.OpenGroup(name)
            if grp is None:
                return None
        ar = grp.OpenMDArray(parts[-1])
        if ar is None:
            return None
        val = _read_mdarray_value(ar)
    except Exception as e:
        logger.debug(f"_read_string_mdarray({filename}, {dset_path}) failed: {e}")
        return None
    finally:
        ar = grp = ds = None

    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, str):
        return val
    return None


def _read_nisar_geoinfo_multidim(
    filename: Filename, dataset_name: str | None = None
) -> tuple[tuple[float, ...], str] | None:
    """Read (geotransform, projection_wkt) from a NISAR GSLC via multidim API.

    Uses GDAL's multidimensional raster API to read xCoordinates/yCoordinates
    and the projection's epsg_code attribute. Works over /vsis3 (range reads).
    Cached so repeated callers for the same file pay only one round trip.
    """
    if not HAS_GDAL:
        return None
    path = fspath(filename)
    freq = "A"
    if dataset_name and "/frequency" in dataset_name:
        after = dataset_name.split("/frequency", 1)[1]
        if after and after[0] in ("A", "B"):
            freq = after[0]
    return _read_nisar_geoinfo_multidim_cached(path, freq)


@functools.lru_cache(maxsize=256)
def _read_nisar_geoinfo_multidim_cached(
    path: str, freq: str
) -> tuple[tuple[float, ...], str] | None:

    ds = rg = grp = None
    try:
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            return None
        rg = ds.GetRootGroup()
        grp = rg
        for name in ("science", "LSAR", "GSLC", "grids"):
            grp = grp.OpenGroup(name)
            if grp is None:
                return None
        grp = grp.OpenGroup(f"frequency{freq}")
        if grp is None:
            return None
        f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        x = grp.OpenMDArray("xCoordinates").ReadAsArray(buffer_datatype=f64)
        y = grp.OpenMDArray("yCoordinates").ReadAsArray(buffer_datatype=f64)
        wkt = _read_nisar_projection_wkt(grp.OpenMDArray("projection"))
    except Exception as e:
        logger.debug(f"_read_nisar_geoinfo_multidim failed for {path}: {e}")
        return None
    finally:
        grp = rg = ds = None

    if x is None or y is None or x.size < 2 or y.size < 2:
        return None
    if not wkt:
        return None
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    gt = (float(x[0]), dx, 0.0, float(y[0]), 0.0, dy)
    return gt, wkt


def _read_nisar_projection_wkt(proj_ar) -> str:
    """Get WKT for a NISAR `projection` MDArray (via `spatial_ref` attribute)."""
    try:
        sr = proj_ar.GetAttribute("spatial_ref")
        if sr is not None:
            wkt = _coerce_wkt(sr.Read())
            if wkt:
                return wkt
    except Exception:
        pass
    # Fallback: build WKT from the integer EPSG code
    for getter in (
        lambda: proj_ar.GetAttribute("epsg_code").Read(),
        lambda: proj_ar.ReadAsArray().item(),
    ):
        try:
            val = getter()
        except Exception:
            continue
        try:
            epsg = int(val if not isinstance(val, (list, tuple)) else val[0])
        except (TypeError, ValueError):
            continue
        srs = osr.SpatialReference()
        if srs.ImportFromEPSG(epsg) == 0:
            return srs.ExportToWkt()
    return ""


_WKT_PREFIXES = ("PROJCS", "GEOGCS", "PROJCRS", "GEOGCRS", "COMPD_CS", "LOCAL_CS")


def _coerce_wkt(raw) -> str:
    """Coerce whatever GDAL's multidim Read() returns into a WKT string.

    Accepts str / bytes / bytearray / nested lists / tuples / numpy object
    arrays; returns "" if no WKT-looking value is found.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        s = raw.strip().strip("\x00").strip()
        return s if s.upper().startswith(_WKT_PREFIXES) else ""
    if isinstance(raw, (bytes, bytearray)):
        try:
            return _coerce_wkt(bytes(raw).decode("utf-8", errors="replace"))
        except Exception:
            return ""
    if isinstance(raw, np.ndarray):
        for item in raw.ravel():
            got = _coerce_wkt(item)
            if got:
                return got
        return ""
    if isinstance(raw, (list, tuple)):
        for item in raw:
            got = _coerce_wkt(item)
            if got:
                return got
        return ""
    return ""
