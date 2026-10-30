"""Feature definitions for the UCI Mushroom dataset (see data/raw/agaricus-lepiota.names)."""

UNKNOWN = "?"

# column -> (display label, {code: display name}), in file order.
FEATURES = {
    "cap-shape": ("Cap shape", {"b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat", "k": "Knobbed", "s": "Sunken"}),
    "cap-surface": ("Cap surface", {"f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"}),
    "cap-color": ("Cap color", {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "r": "Green", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"}),
    "bruises": ("Bruises", {"t": "Yes", "f": "No"}),
    "odor": ("Odor", {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"}),
    "gill-attachment": ("Gill attachment", {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"}),
    "gill-spacing": ("Gill spacing", {"c": "Close", "w": "Crowded", "d": "Distant"}),
    "gill-size": ("Gill size", {"b": "Broad", "n": "Narrow"}),
    "gill-color": ("Gill color", {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Gray", "r": "Green", "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"}),
    "stalk-shape": ("Stalk shape", {"e": "Enlarging", "t": "Tapering"}),
    "stalk-root": ("Stalk root", {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs", "r": "Rooted"}),
    "stalk-surface-above-ring": ("Stalk surface above ring", {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"}),
    "stalk-surface-below-ring": ("Stalk surface below ring", {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"}),
    "stalk-color-above-ring": ("Stalk color above ring", {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"}),
    "stalk-color-below-ring": ("Stalk color below ring", {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"}),
    "veil-type": ("Veil type", {"p": "Partial", "u": "Universal"}),
    "veil-color": ("Veil color", {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"}),
    "ring-number": ("Number of rings", {"n": "None", "o": "One", "t": "Two"}),
    "ring-type": ("Ring type", {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone"}),
    "spore-print-color": ("Spore print color", {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"}),
    "population": ("Population", {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"}),
    "habitat": ("Habitat", {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste", "d": "Woods"}),
}

COLUMNS = ["class", *FEATURES]
