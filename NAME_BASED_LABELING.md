# Name-Based Class Labeling System

## Overview
The data synthesis application now implements a **name-based class labeling system** that ensures consistent class assignment across all candidate objects based on their names.

## Key Features

### 1. Consistent Class Assignment
- **Same Name = Same Label**: All objects in the candidate list with the exact same name will receive the same `class_id`
- **Different Names = Different Labels**: Objects with different names will get unique `class_id` values
- **Independent of Creation Order**: Class IDs are assigned during generation, not during object creation

### 2. Color Consistency
- **Bounding Box Colors**: Each class gets a unique color for bounding box visualization
- **Segmentation Colors**: The same class uses the same color in segmentation masks
- **Visual Differentiation**: Different labels have visually distinct colors for easy identification

## Implementation Details

### Modified Methods

#### 1. `_generate_annotations()`
**Location**: `data_synthesis_app.py` line ~2214

**Changes**:
- Creates a `name_to_class_id` mapping at the start of annotation generation
- Iterates through all candidate objects to build the mapping
- Uses candidate name (not object's original class_id) to determine the class
- Ensures same names get same class_id across all instances

```python
# Build name-based class mapping from candidate objects
name_to_class_id = {}
class_id_counter = 0

for candidate in self.candidate_objects:
    candidate_name = candidate["name"]
    if candidate_name not in name_to_class_id:
        name_to_class_id[candidate_name] = class_id_counter
        class_id_counter += 1

# Later in the loop:
candidate_name = candidate["name"]
class_id = name_to_class_id[candidate_name]
class_name = candidate_name
```

#### 2. `_create_bbox_image()`
**Location**: `data_synthesis_app.py` line ~2006

**Changes**:
- Builds the same `name_to_class_id` mapping before rendering
- Uses mapped class_id to select bbox color
- Ensures consistent colors: same name = same bbox color

```python
# Build name-based class mapping
name_to_class_id = {}
class_id_counter = 0

for candidate in self.candidate_objects:
    candidate_name = candidate["name"]
    if candidate_name not in name_to_class_id:
        name_to_class_id[candidate_name] = class_id_counter
        class_id_counter += 1

# Use mapped class_id for color
candidate_name = candidate["name"]
class_id = name_to_class_id[candidate_name]
class_name = candidate_name
color = class_colors[class_id % len(class_colors)]
```

#### 3. `_create_segmentation_visualization()`
**Location**: `data_synthesis_app.py` line ~2143

**Changes**:
- Builds the same `name_to_class_id` mapping
- Uses mapped class_id to select segmentation color
- Ensures segmentation and bbox use the same class_id for the same object name

```python
# Build name-based class mapping
name_to_class_id = {}
class_id_counter = 0

for candidate in self.candidate_objects:
    candidate_name = candidate["name"]
    if candidate_name not in name_to_class_id:
        name_to_class_id[candidate_name] = class_id_counter
        class_id_counter += 1

# Use mapped class_id for segmentation color
candidate_name = candidate["name"]
class_id = name_to_class_id[candidate_name]
seg_color = np.array(class_colors[class_id % len(class_colors)], dtype=np.float32)
```

## Usage Examples

### Example 1: Multiple Cats
If you have 3 cat objects in your candidate list:
- `cat` (instance 1)
- `cat` (instance 2)
- `cat` (instance 3)

**Result**:
- All 3 will get `class_id = 0`
- All 3 will have the same bbox color (e.g., Red)
- All 3 will have the same segmentation color
- COCO/YOLO annotations will all use class name "cat" with id 0

### Example 2: Mixed Objects
If you have:
- `cat` (instance 1)
- `dog` (instance 1)
- `cat` (instance 2)
- `bird` (instance 1)

**Result**:
- Both cats: `class_id = 0`, Red color
- Dog: `class_id = 1`, Green color
- Bird: `class_id = 2`, Blue color
- Same names share colors, different names have distinct colors

### Example 3: Group Objects
If you add a group called "furniture" containing:
- `chair` (child 1)
- `table` (child 1)
- `chair` (child 2)

**Result**:
- Both chairs: `class_id = 0`, Red color
- Table: `class_id = 1`, Green color

## Color Palette

The system uses 10 predefined colors that cycle for classes:
1. Red (255, 0, 0)
2. Green (0, 255, 0)
3. Blue (0, 0, 255)
4. Yellow (255, 255, 0)
5. Magenta (255, 0, 255)
6. Cyan (0, 255, 255)
7. Orange (255, 128, 0)
8. Purple (128, 0, 255)
9. Spring Green (0, 255, 128)
10. Rose (255, 0, 128)

If you have more than 10 different object names, colors will cycle using modulo: `class_id % 10`

## Benefits

1. **Semantic Consistency**: Objects are grouped by what they are (their name), not when they were created
2. **Intuitive Training Data**: ML models will learn that all instances of "cat" belong to the same class
3. **Visual Clarity**: Easy to identify which objects belong to the same class during data review
4. **Flexible Workflow**: You can create objects at different times and add them to candidates in any order

## Technical Notes

- The mapping is rebuilt for each generation, ensuring consistency even if candidates are modified
- Class IDs are assigned in the order names first appear in the candidate list
- The system is independent of the object's original `class_id` from scene creation
- All three visualization types (annotations, bbox, segmentation) use the same mapping algorithm
