import numpy as np
import math

# WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])
WORKSPACE_LIMITS = np.asarray([[0.22, 0.78], [-0.28, 0.28], [-0.0001, 0.4]])

# object
OBJECT_SYMMETRIC_MAP = {
    "000": False,
    "001": False,
    "002": False,
    "003": False,
    "004": False,
    "005": False,
    "006": True,
    "007": False,
    "008": False,
    "009": False,
    "010": False,
    "011": False,
    "012": True,
    "013": False,
    "014": False,
    "015": True,
    "016": True,
    "017": True,
    "018": False,
    "019": False,
    "020": True,
    "021": True,
    "022": True,
    "023": True,
    "024": False,
    "025": False,
    "026": True,
    "027": True,
    "028": True,
    "029": True,
    "030": True,
    "031": True,
    "032": False,
    "033": False,
    "034": False,
    "035": False,
    "036": False,
    "037": False,
    "038": False,
    "039": True,
    "040": False,
    "041": False,
    "042": False,
    "043": False,
    "044": False,
    "045": False,
    "046": False,
    "047": False,
    "049": False,
    "050": False,
    "052": False,
    "053": False,
    "055": False,
    "057": False,
    "058": False,
    "059": False,
    "061": False,
    "062": False,
    "063": False,
    "064": False,
    "065": False,
    "066": False,
    "067": False,
    "068": False,
    "070": True,
    "072": True,
    "073": True,
    "074": False,
    "075": False,
    "076": False,
    "077": False,
    "078": False,
    "079": False,
    "080": False,
    "081": False,
    "082": False,
    "083": False,
    "084": False,
    "085": False,
    "086": False,
    "087": False,
}

UNSEEN_OBJECT_SYMMETRIC_MAP = {
    "000": False,
    "001": False,
    "003": False,
    "004": False,
    "006": True,
    "010": False,
    "018": False,
    "019": False,
    "023": True,
    "025": False,
    "033": False,
    "035": False,
    "036": False,
    "045": True,
    "046": False,
    "049": False,
    "050": False,
    "055": False,
    "063": False,
    "065": False,
    "067": False,
    "072": True,
    "073": True,
    "075": False,
}

# image
# PIXEL_SIZE = 0.002
PIXEL_SIZE = 0.0025
IMAGE_SIZE = 224
IMAGE_OBJ_CROP_SIZE = 60  # this is related to the IMAGE_SIZE and PIXEL_SIZE
IMAGE_PAD_SIZE = math.ceil(IMAGE_SIZE * math.sqrt(2) / 32) * 32  # 320
IMAGE_PAD_WIDTH = math.ceil((IMAGE_PAD_SIZE - IMAGE_SIZE) / 2)  # 48
IMAGE_PAD_DIFF = IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH  # 272

# gripper
GRIPPER_GRASP_INNER_DISTANCE = 0.07
GRIPPER_GRASP_INNER_DISTANCE_PIXEL = math.ceil(GRIPPER_GRASP_INNER_DISTANCE / PIXEL_SIZE)  # 40
GRIPPER_GRASP_OUTER_DISTANCE = 0.125
GRIPPER_GRASP_OUTER_DISTANCE_PIXEL = math.ceil(GRIPPER_GRASP_OUTER_DISTANCE / PIXEL_SIZE)  # 63
GRIPPER_GRASP_WIDTH = 0.022
GRIPPER_GRASP_WIDTH_PIXEL = math.ceil(GRIPPER_GRASP_WIDTH / PIXEL_SIZE)  # 11
GRIPPER_GRASP_SAFE_WIDTH = 0.025
GRIPPER_GRASP_SAFE_WIDTH_PIXEL = math.ceil(GRIPPER_GRASP_SAFE_WIDTH / PIXEL_SIZE)  # 13

KNOWN_OBJ_IDS = ["000", "002", "005", "007", "008", "009", "011", "014", "017", "018", "020", "021",
	"022", "026", "027", "029", "030", "038", "041", "048", "051", "052", "058", "060",
	"061", "062", "063", "066"]

NUM_POINTS = 10000
TABLE_WIDTH = 0.448
TABLE_LENGTH = 0.448
FIX_MAX_ERR = np.linalg.norm(np.array([TABLE_LENGTH, TABLE_WIDTH])) / 2.0