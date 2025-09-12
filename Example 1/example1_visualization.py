import numpy as np
import pylab, ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def conditional_round(number, tolerance=1e-6):
    """
    Conditionally rounds a number to the nearest integer if it is within a specified tolerance.

    Parameters:
        number (float): The number to potentially round.
        tolerance (float): The maximum allowed difference from the nearest integer to perform rounding.

    Returns:
        float: The rounded number if within tolerance, otherwise the original number.
    """
    rounded = round(number)
    # Check if the number is within the tolerance of its rounded value
    if abs(number - rounded) <= tolerance:
        return rounded
    else:
        return number

def find_closest_element(sample, lst):
    """
    Finds the element in a list that is closest to a given sample value and returns it along with its index.

    Parameters:
        sample (float or complex): The reference value to compare against.
        lst (list of float or complex): The list of elements to search.

    Returns:
        tuple: A tuple containing the closest element and its index in the list.
    """
    # Convert the list to a NumPy array for efficient computation
    lst_array = np.asarray(lst)
    # Compute the absolute differences between the sample and each element in the list
    differences = np.abs(lst_array - sample)
    # Find the index of the minimum difference
    index = np.argmin(differences)
    # Retrieve the closest element
    closest_element = lst_array[index]
    return closest_element, index

def are_close(a, b, tol=1e-8):
    """
    Determines whether two numbers are close to each other within a specified tolerance.

    Parameters:
        a (float or complex): The first number.
        b (float or complex): The second number.
        tol (float): The maximum allowed difference to consider the numbers as close.

    Returns:
        bool: True if the numbers are within the specified tolerance, False otherwise.
    """
    # Compute the absolute difference and compare with the tolerance
    return np.abs(a - b) <= tol

def sin(x):
    """
    Calculates the sine of an angle provided in degrees.

    Parameters:
        x (float): The angle in degrees.

    Returns:
        float: The sine of the angle.
    """
    # Convert angle from degrees to radians and compute sine
    radians = np.deg2rad(x, dtype='float64')
    return np.sin(radians, dtype='float64')

def cos(x):
    """
    Calculates the cosine of an angle provided in degrees.

    Parameters:
        x (float): The angle in degrees.

    Returns:
        float: The cosine of the angle.
    """
    # Convert angle from degrees to radians and compute cosine
    radians = np.deg2rad(x)
    return np.cos(radians)

def tan(x):
    """
    Calculates the tangent of an angle provided in degrees.

    Parameters:
        x (float): The angle in degrees.

    Returns:
        float: The tangent of the angle.
    """
    # Convert angle from degrees to radians and compute tangent
    radians = np.deg2rad(x, dtype='float64')
    return np.tan(radians, dtype='float64')

def arctan(x):
    """
    Computes the arctangent of a value and returns the result in degrees.

    Parameters:
        x (float): The value to compute the arctangent for.

    Returns:
        float: The arctangent of x in degrees.
    """
    # Calculate arctangent in radians with double precision
    atan_rad = np.arctan(x, dtype='float64')
    # Convert the result from radians to degrees
    atan_deg = np.degrees(atan_rad, dtype='float64')
    return atan_deg


def generate_edge(A1, angle1, angle2, angle3, booster=1.0):
    """
    Calculates the length of edge A2 based on provided angles and the length of edge A1,
    ensuring the quadrilateral configuration satisfies geometric constraints.

    This function uses trigonometric relationships to compute A2, considering conditions
    derived from geometric requirements (e.g., positive edge lengths). The calculation
    adapts based on the sum of input angles and applies a booster factor to adjust the result.

    Parameters:
        A1 (float): Length of the first edge (A1A2).
        angle1 (float): First angle in degrees (associated with A1).
        angle2 (float): Second angle in degrees.
        angle3 (float): Third angle in degrees.
        booster (float, optional): A scaling factor to adjust A2. Default is 1.0.

    Returns:
        float: The calculated length of edge A2.

    Raises:
        ValueError: If the sum of angles leads to invalid computations.

    Notes:
        - Conditions are derived from the requirement that edge lengths a3 > 0 and a4 > 0.
          This ensures the quadrilateral is valid.
        - For the given quadrilateral A1A2A3A4:
            - `delta1` is angle A4A1A2,
            - `delta2` is angle A1A2A3,
            - `delta3` is angle A2A3A4,
            - `delta4` is angle A3A4A1.
        - Angles and edges are taken in the same order as `delta1`, `delta2`, `delta3`, where:
            - `angle1` corresponds to `delta1`,
            - `angle2` corresponds to `delta2`,
            - `angle3` corresponds to `delta3`.
        - Edges are labeled as:
            - `a1` = A1A2 (length A1),
            - `a2` = A2A3 (length A2),
            - corresponding to the quadrilateral A1A2A3A4.
        - For detailed derivations and geometric constraints, refer to the reference PDF.
    """
    # Ensure inputs are float64 for precision
    A1 = np.float64(A1)
    angle1 = np.float64(angle1)
    angle2 = np.float64(angle2)
    angle3 = np.float64(angle3)
    booster = np.float64(booster)

    # Calculate sums of angles for condition checks
    sum_angle2_3 = angle2 + angle3
    sum_angle1_2 = angle1 + angle2

    # Initialize A2
    A2 = None

    if sum_angle2_3 >= 180.0:
        # Case when the sum of angle2 and angle3 is greater than or equal to 180 degrees
        if sum_angle1_2 >= 180.0:
            # Both angle sums are greater than or equal to 180 degrees
            A2 = booster * A1
            # Debug output when booster is 1000
            if booster == 1000.0:
                print("Choose A2 as you wish, i.e. choose booster as you wish.")
        else:
            # Calculate A2 using the sine rule when angle1 + angle2 < 180
            numerator = booster * 0.8 * A1 * sin(angle1)
            denominator = sin(sum_angle1_2)
            A2 = numerator / denominator
            # Debug output when booster is 1000
            if booster == 1000.0:
                print("Case 0.8: Choose booster strictly less than 5/4.")
    else:
        # Case when the sum of angle2 and angle3 is less than 180 degrees
        if sum_angle1_2 >= 180.0:
            # Calculate A2 using the sine rule when angle1 + angle2 >= 180
            numerator = booster * 1.2 * A1 * sin(sum_angle2_3)
            denominator = sin(angle3)
            A2 = numerator / denominator
            # Debug output when booster is 1000
            if booster == 1000.0:
                print("Case 1.2: Choose booster strictly greater than 5/6.")
        else:
            # Compare two scale factors based on sine ratios: sin(angle1) / sin(angle1 + angle2) and sin(angle2 + angle3) / sin(angle3)
            left_scale = sin(sum_angle2_3) / sin(angle3)
            right_scale = sin(angle1) / sin(sum_angle1_2)

            # Determine scale ratio based on the comparison of the two sine ratios
            if left_scale > right_scale:
                scale_ratio = left_scale // right_scale + 1
            elif right_scale > left_scale:
                scale_ratio = right_scale // left_scale + 1
            else:
                scale_ratio = 2.0

            # Debug output when booster is 1000
            if booster == 1000.0:
                print("scaleRatio:", scale_ratio, f"Choose booster between 0 and {scale_ratio}.")

            # Compute A2 using a weighted average of the two scales based on booster and scaleRatio
            A2 = A1 * ((1.0 - booster / scale_ratio) * left_scale + (booster / scale_ratio) * right_scale)

    # Check if A2 was calculated successfully
    if A2 is None:
        raise ValueError("Failed to compute A2 with the given angles and booster.")

    return A2

def generate_tilde_angle(angle1, angle2, booster=1.0):
    """
    Calculates a tilde angle based on two input angles and an optional booster value,
    ensuring that the resulting angles satisfy geometric constraints.

    The function computes the tilde angle using a scaling factor `k`, such that:
        angle3 = k * (360 - angle1 - angle2)
        angle4 = (1 - k) * (360 - angle1 - angle2)
    The computed angles ensure that 0 < angle3 < 180 and 0 < angle4 < 180 degrees.

    Parameters:
        angle1 (float): The first angle in degrees.
        angle2 (float): The second angle in degrees.
        booster (float, optional): A scaling factor to adjust the tilde angle. Default is 1.0.

    Returns:
        float: The generated tilde angle (angle3 in degrees).

    Raises:
        ValueError: If the sum of angle1 and angle2 is invalid.
    """
    # Ensure inputs are float64 for precision
    angle1 = np.float64(angle1)
    angle2 = np.float64(angle2)
    booster = np.float64(booster)

    sum_angle1_2 = angle1 + angle2

    # Check for invalid input where the sum of angles is 360 degrees or more
    if sum_angle1_2 >= 360.0:
        raise ValueError("Invalid angles: The sum of angle1 and angle2 must be less than 360 degrees.")

    remaining_angles = 360.0 - sum_angle1_2

    if sum_angle1_2 >= 180.0:
        # Case when the sum of angle1 and angle2 is 180 degrees or more
        denominator = remaining_angles
        # Calculate scaling factor k
        k = min(booster * 0.5 * 180.0 / denominator, 0.5)
        # Debug output when booster is 1000
        if booster == 1000.0:
            print("Case 1: Sum of angles >= 180. Choose booster strictly positive.")
            print("k =", k)
    else:
        # Case when the sum of angle1 and angle2 is less than 180 degrees
        numerator = (1.0 - 0.6 * booster) * (180.0 - sum_angle1_2) + 0.6 * booster * 180.0
        denominator = remaining_angles
        # Calculate scaling factor k
        k = numerator / denominator
        # Debug output when booster is 1000
        if booster == 1000.0:
            print("Case 2: Sum of angles < 180. Choose booster between 0 and 5/3.")
            print("k =", k)

    # Compute the tilde angle (angle3)
    angle3 = remaining_angles * k

    return angle3

def convert_angles_to_vertices(Alphas, Gammas, Deltas, DihedralAngles, Edge):
    """
    Converts given angles and edge lengths into 3D vertex coordinates for a quadrilateral structure.

    The quadrilateral A1A2A3A4 lies in the xy-plane, with edge A2A1 aligned along the x-axis.
    The y-axis forms an acute angle with edge A2A3. Faces 1, 2, 3, and 4 are oriented clockwise,
    while face C is oriented counterclockwise. The vertices are calculated based on the provided
    angles and edge lengths using trigonometric relationships.

    Parameters:
        Alphas (list of float): A list of four alpha angles [alpha1, alpha2, alpha3, alpha4] in degrees.
        Gammas (list of float): A list of four gamma angles [gamma1, gamma2, gamma3, gamma4] in degrees.
        Deltas (list of float): A list of four delta angles [delta1, delta2, delta3, delta4] in degrees.
        DihedralAngles (list of float): A list of four dihedral angles [phi, psi2, theta, psi1] in degrees.
        Edge (float): The length of edge A1A2.

    Returns:
        dict: A dictionary containing vertex coordinates labeled as A1, A2, A3, A4, B1, B2, B3, B4,
              C1, C2, C3, C4, with each value being a tuple of (x, y, z) coordinates.

    Notes:
        - The vertices are calculated using trigonometric functions, ensuring that all angles are in degrees.
        - All computations are performed using float64 precision to maintain numerical accuracy.
        - The function assumes that the input angles and edge lengths form a valid geometric configuration.
        - Refer to the reference PDF for detailed derivations and geometric relationships.

    Raises:
        ValueError: If any of the trigonometric computations result in invalid values due to incorrect inputs.
    """
    # Ensure inputs are numpy float64 for precision
    alpha1, alpha2, alpha3, alpha4 = map(np.float64, Alphas)
    gamma1, gamma2, gamma3, gamma4 = map(np.float64, Gammas)
    delta1, delta2, delta3, delta4 = map(np.float64, Deltas)
    phi, psi2, theta, psi1 = map(np.float64, DihedralAngles)
    A1A2 = np.float64(Edge)
    
    # Initialize the dictionary to store the vertices
    vertices = {}

    # -------------------- Plane C: Calculate vertices A1, A2, A3, A4 --------------------
    # Edge lengths
    a1 = A1A2  # Edge length A1A2
    # Generate edge A2A3 using the provided function with a booster factor
    a2 = generate_edge(a1, delta1, delta2, delta3, booster=1)  # Edge length A2A3
    
    # print(a2)

    # Coordinates of vertex A1 (lies on the x-axis)
    xA1, yA1, zA1 = a1, 0.0, 0.0
    # Coordinates of vertex A2 (origin)
    xA2, yA2, zA2 = 0.0, 0.0, 0.0

    # Calculate coordinates of vertex A3
    xA3 = xA2 + a2 * cos(delta2)
    yA3 = yA2 + a2 * sin(delta2)
    zA3 = zA2

    # Calculate edge length A1A4 (edge a3) using trigonometric relationships
    A1A4 = (a2 * sin(delta3) - a1 * sin(delta2 + delta3)) / sin(delta4) 
    a3 = A1A4 # Edge length A1A4
    
    # Calculate edge length A3A4 (edge a4) using trigonometric relationships
    A3A4 = (a1 * sin(delta1) - a2 * sin(delta1 + delta2)) / sin(delta4)
    a4 = A3A4 # Edge length A3A4
    
    # print(a3, a4)
    
    # Calculate coordinates of vertex A4
    xA4 = xA1 - a3 * cos(delta1)
    yA4 = yA1 + a3 * sin(delta1)
    zA4 = zA1

    # -------------------- Plane 1: Calculate vertices B1, B2 --------------------
    # Generate tilde angles for Plane 1
    alpha3_tilde = generate_tilde_angle(alpha1, alpha2, booster=1)  # Angle A2B2B1
    alpha4_tilde = 360.0 - alpha1 - alpha2 - alpha3_tilde  # Angle A1B1B2
    

    # Edge lengths for Plane 1
    A2B2 = generate_edge(a1, alpha1, alpha2, alpha3_tilde, booster=8) 
    b2 = A2B2 # Edge length A2B2

    # Calculate edge length A1B1 (edge b1) using trigonometric relationships
    A1B1 = (b2 * sin(alpha3_tilde) - a1 * sin(alpha2 + alpha3_tilde)) / sin(alpha4_tilde)
    b1 = A1B1 # Edge length A1B1
    
    # print(b2, b1)

    # Calculate coordinates of vertex B1
    xB1 = xA1 - b1 * cos(alpha1)
    yB1 = yA1 + b1 * sin(alpha1) * cos(phi)
    zB1 = zA1 + b1 * sin(alpha1) * sin(phi)

    # Calculate coordinates of vertex B2
    xB2 = xA2 + b2 * cos(alpha2)
    yB2 = yA2 + b2 * sin(alpha2) * cos(phi)
    zB2 = zA2 + b2 * sin(alpha2) * sin(phi)

    # -------------------- Plane 2: Calculate vertices C2, C3 --------------------
    # Generate tilde angles for Plane 2
    gamma1_tilde = generate_tilde_angle(gamma3, gamma2, booster=1)  # Angle A2C2C3
    gamma4_tilde = 360.0 - gamma2 - gamma3 - gamma1_tilde  # Angle A3C3C2
    

    # Edge lengths for Plane 2
    A2C2 = generate_edge(a2, gamma3, gamma2, gamma1_tilde, booster=1.5)
    c2 = A2C2 # Edge length A2C2

    # Calculate edge length A3C3 (edge c3) using trigonometric relationships
    A3C3 = (c2 * sin(gamma1_tilde) - a2 * sin(gamma2 + gamma1_tilde)) / sin(gamma4_tilde)
    c3 = A3C3 # Edge length A3C3
    
    # print(c2, c3)
    
    # Calculate coordinates of vertex C2
    xC2 = xA2 + c2 * (cos(gamma2) * cos(delta2) + sin(gamma2) * cos(psi2) * sin(delta2))
    yC2 = yA2 + c2 * (cos(gamma2) * sin(delta2) - sin(gamma2) * cos(psi2) * cos(delta2))
    zC2 = zA2 + c2 * sin(gamma2) * sin(psi2)

    # Calculate coordinates of vertex C3
    xC3 = xA3 - c3 * (cos(gamma3) * cos(delta2) - sin(gamma3) * cos(psi2) * sin(delta2))
    yC3 = yA3 - c3 * (cos(gamma3) * sin(delta2) + sin(gamma3) * cos(psi2) * cos(delta2))
    zC3 = zA3 + c3 * sin(gamma3) * sin(psi2)

    # -------------------- Plane 3: Calculate vertices B3, B4 --------------------
    # Generate tilde angles for Plane 3
    alpha1_tilde = generate_tilde_angle(alpha4, alpha3, booster=1)  # Angle A3B3B4
    alpha2_tilde = 360.0 - alpha3 - alpha4 - alpha1_tilde  # Angle A4B4B3

    # Edge lengths for Plane 3
    A3B3 = generate_edge(a4, alpha4, alpha3, alpha1_tilde, booster=8)
    b3 = A3B3 # Edge length A3B3

    # Calculate edge length A4B4 (edge b4) using trigonometric relationships
    A4B4 = (b3 * sin(alpha1_tilde) - a4 * sin(alpha3 + alpha1_tilde)) / sin(alpha2_tilde)
    b4 = A4B4 # Edge length A4B4
    
    # print(b3, b4)
    
    # Calculate coordinates of vertex B3
    sum_delta2_3 = delta2 + delta3
    xB3 = xA3 - b3 * (cos(alpha3) * cos(sum_delta2_3) + sin(alpha3) * cos(theta) * sin(sum_delta2_3))
    yB3 = yA3 - b3 * (cos(alpha3) * sin(sum_delta2_3) - sin(alpha3) * cos(theta) * cos(sum_delta2_3))
    zB3 = zA3 + b3 * sin(alpha3) * sin(theta)

    # Calculate coordinates of vertex B4
    xB4 = xA4 + b4 * (cos(alpha4) * cos(sum_delta2_3) - sin(alpha4) * cos(theta) * sin(sum_delta2_3))
    yB4 = yA4 + b4 * (cos(alpha4) * sin(sum_delta2_3) + sin(alpha4) * cos(theta) * cos(sum_delta2_3))
    zB4 = zA4 + b4 * sin(alpha4) * sin(theta)

    # -------------------- Plane 4: Calculate vertices C1, C4 --------------------
    # Generate tilde angles for Plane 4
    gamma2_tilde = generate_tilde_angle(gamma1, gamma4, booster=1)  # Angle A4C4C1
    gamma3_tilde = 360.0 - gamma1 - gamma4 - gamma2_tilde  # Angle A1C1C4
    

    # Edge lengths for Plane 4
    A4C4 = generate_edge(a3, gamma1, gamma4, gamma2_tilde, booster=1.6)
    c4 = A4C4 # Edge length A4C4

    # Calculate edge length A1C1 (edge c1) using trigonometric relationships
    A1C1 = (c4 * sin(gamma2_tilde) - a3 * sin(gamma4 + gamma2_tilde)) / sin(gamma3_tilde)
    c1 = A1C1 # Edge length A1C1
    
    print(c4, c1)
    
    
    # Calculate coordinates of vertex C4
    xC4 = xA4 + c4 * (cos(gamma4) * cos(delta1) - sin(gamma4) * cos(psi1) * sin(delta1))
    yC4 = yA4 - c4 * (cos(gamma4) * sin(delta1) + sin(gamma4) * cos(psi1) * cos(delta1))
    zC4 = zA4 + c4 * sin(gamma4) * sin(psi1)

    # Calculate coordinates of vertex C1
    xC1 = xA1 - c1 * (cos(gamma1) * cos(delta1) + sin(gamma1) * cos(psi1) * sin(delta1))
    yC1 = yA1 + c1 * (cos(gamma1) * sin(delta1) - sin(gamma1) * cos(psi1) * cos(delta1))
    zC1 = zA1 + c1 * sin(gamma1) * sin(psi1)

    # -------------------- Store the vertices in the dictionary --------------------
    vertices = {
        r"$A_1$": (xA1, yA1, zA1),
        r"$A_2$": (xA2, yA2, zA2),
        r"$A_3$": (xA3, yA3, zA3),
        r"$A_4$": (xA4, yA4, zA4),
        r"$B_1$": (xB1, yB1, zB1),
        r"$B_2$": (xB2, yB2, zB2),
        r"$B_3$": (xB3, yB3, zB3),
        r"$B_4$": (xB4, yB4, zB4),
        r"$C_1$": (xC1, yC1, zC1),
        r"$C_2$": (xC2, yC2, zC2),
        r"$C_3$": (xC3, yC3, zC3),
        r"$C_4$": (xC4, yC4, zC4),
    }

    return vertices

def visualize_here(Alphas, Betas, Gammas, Deltas, Edge):
    """
    Visualizes 3D vertices and faces using Matplotlib, allowing interactive adjustments.

    This function creates a 3D plot of the given vertices and connects them to form faces.
    Each vertex is labeled, and the 3D model can be rotated with mouse events. A slider
    allows adjusting the dihedral angle phi, and the plot updates accordingly to reflect
    changes in the geometry.

    Parameters:
        Alphas (list of float): List of alpha angles [alpha1, alpha2, alpha3, alpha4] in degrees.
        Betas (list of float): List of beta angles [beta1, beta2, beta3, beta4] in degrees.
        Gammas (list of float): List of gamma angles [gamma1, gamma2, gamma3, gamma4] in degrees.
        Deltas (list of float): List of delta angles [delta1, delta2, delta3, delta4] in degrees.
        Edge (float): Length of the edge A1A2.

    Notes:
        - The function relies on the global variable DihedralAngles_Init for initial dihedral angles.
        - The function uses custom trigonometric functions and helper functions defined elsewhere.
        - The 3D plot includes interactive elements such as sliders and buttons for user interaction.
        - Commented-out code is preserved and updated to provide context and alternative visualization options.
    """
    # Initialize dihedral angles from global variable
    phi_init, psi2_init, theta_init, psi1_init = map(np.float64, DihedralAngles_Init)
    DihedralAngles = [phi_init, psi2_init, theta_init, psi1_init]

    # Unpack angles
    alpha1, alpha2, alpha3, alpha4 = map(np.float64, Alphas)
    beta1, beta2, beta3, beta4 = map(np.float64, Betas)
    gamma1, gamma2, gamma3, gamma4 = map(np.float64, Gammas)
    delta1, delta2, delta3, delta4 = map(np.float64, Deltas)

    # Compute initial vertices
    Vertices = convert_angles_to_vertices(Alphas, Gammas, Deltas, DihedralAngles, Edge)

    # Prepare vertices and labels
    vertices_list = [list(val) for val in Vertices.values()]
    vertices_labels = list(Vertices.keys())

    # Define faces (polygon surfaces) using vertex indices
    Faces = [
        [0, 1, 2, 3],       # Face A1-A2-A3-A4
        [0, 4, 5, 1],       # Face A1-B1-B2-A2
        [1, 9, 10, 2],      # Face A2-C2-C3-A3
        [2, 6, 7, 3],       # Face A3-B3-B4-A4
        [3, 11, 8, 0]       # Face A4-C4-C1-A1
    ]

    AllFaces = [
        [0, 1, 2, 3],       # Face A1-A2-A3-A4
        [0, 4, 5, 1],       # Face A1-B1-B2-A2
        [1, 5, 9],          # Face A2-B2-C2
        [1, 9, 10, 2],      # Face A2-C2-C3-A3
        [2, 10, 6],         # Face A3-C3-B3
        [2, 6, 7, 3],       # Face A3-B3-B4-A4
        [3, 7, 11],         # Face A4-B4-C4
        [3, 11, 8, 0],      # Face A4-C4-C1-A1
        [0, 8, 4]           # Face A1-C1-B1
    ]

    # Define face colors
    Colors = ['slategrey', 'crimson', 'slateblue', 'turquoise', 'orangered']
    AllColors = [
        'slategrey', 'crimson', 'slateblue', 'turquoise', 'orangered',
        'seagreen', 'purple', 'royalblue', 'teal'
    ]

    # Map vertex indices to their coordinates
    EdgesToDraw = [[vertices_list[i] for i in face] for face in Faces]
    AllEdgesToDraw = [[vertices_list[i] for i in face] for face in AllFaces]

    # Initialize the 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()  # Turn off the axes
    ax.set(xlim=(-6, 10), ylim=(-6, 10), zlim=(-6, 10))  # Set axis limits

    # Create the 3D polygon collection for the faces
    # Uncomment one of the following lines to change visualization style
    # collection = Poly3DCollection(EdgesToDraw, color=Colors, linewidths=1.1, alpha=0.05)
    # collection = Poly3DCollection(AllEdgesToDraw, color=AllColors, linewidths=1, alpha=0.05)
    # collection = Poly3DCollection(EdgesToDraw, edgecolor='darkblue', linewidths=1.5, alpha=0.5)
    collection = Poly3DCollection(AllEdgesToDraw, edgecolor='darkblue', linewidths=1.5, alpha=0.5)
    ax.add_collection3d(collection)  # Add the collection to the plot

    # Store the number of vertices and initialize labels and positions for annotation
    num_vertices = len(vertices_list)
    labels_list = [None] * num_vertices
    x2d_list, y2d_list = [0] * num_vertices, [0] * num_vertices

    scatter_plots = []
    # Add labels and scatter plots for each vertex
    for i in range(num_vertices):
        # Project 3D coordinates to 2D for annotation placement
        x2d_list[i], y2d_list[i], _ = proj3d.proj_transform(
            vertices_list[i][0], vertices_list[i][1], vertices_list[i][2], ax.get_proj()
        )
        # Create annotations for each vertex
        labels_list[i] = pylab.annotate(
            f"{vertices_labels[i]}", xy=(x2d_list[i], y2d_list[i]), xytext=(0, 0),
            color='darkblue', fontsize=10, textcoords='offset points',
            ha='right', va='bottom',
        )
        labels_list[i].draggable()  # Make labels draggable
        # Plot vertices as scatter points
        scatter_plot = ax.scatter(
            vertices_list[i][0], vertices_list[i][1], vertices_list[i][2],
            color='darkblue', s=30
        )
        scatter_plots.append(scatter_plot)    
    
    
    # Add labels to edges
    # Define edges explicitly using vertex indices
    Edges = [
        [0, 1],  # A1-A2
        [1, 2],  # A2-A3
        [2, 3],  # A3-A4
        [3, 0],  # A4-A1
        [0, 4],  # A1-B1
        [1, 5],  # A2-B2
        [2, 6],  # A3-B3
        [3, 7],  # A4-B4
        [0, 8],  # A1-C1
        [1, 9],  # A2-C2
        [2, 10], # A3-C3
        [3, 11], # A4-C4
        [4, 5],  # B1-B2
        [4, 8],  # B1-C1 
        [6, 7],  # B3-B4 
        [5, 9],  # B2-C2 
        [6, 10], # B3-C3
        [9, 10], # C2-C3 
        [7, 11], # B4-C4
        [11, 8]  # C4-C1  
    ]
    
    # Add labels to edges (lengths formatted with 3 significant digits)
    edge_label_list = []
    for edge in Edges:
        # Extract the edge's vertices from the vertices_list
        edge_vertices = [vertices_list[i] for i in edge]
        # Calculate the length of the edge
        edge_length = np.linalg.norm(np.array(edge_vertices[0]) - np.array(edge_vertices[1]))
        # Format the edge length to 3 significant digits
        edge_label_text = f"${edge_length:.4g}$"
        # Calculate the midpoint of each edge for annotation
        midpoint = np.mean(edge_vertices, axis=0)
        # Project 3D midpoint coordinates to 2D for annotation placement
        x2d_mid, y2d_mid, _ = proj3d.proj_transform(midpoint[0], midpoint[1], midpoint[2], ax.get_proj())
        # Create an annotation for each edge
        edge_label = pylab.annotate(
            edge_label_text, xy=(x2d_mid, y2d_mid), xytext=(0, 0),
            color='red', fontsize=6, textcoords='offset points',
            ha='center', va='center'
        )
        edge_label.draggable()  # Make edge labels draggable
        edge_label_list.append(edge_label)
        
    # Function to calculate the angle between two vectors
    def calculate_angle(v1, v2):
        """Calculates the angle between two vectors in degrees."""
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Ensure the cosine value is in the range [-1, 1] to avoid domain errors in arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        return np.rad2deg(angle_rad)

    # Define which pairs of edges represent each angle (e.g., alpha1, beta1, etc.)
    angles_to_edges = {
        r'$\alpha_1$': ([0, 1], [0, 4]),  # Edges A1A2 and A1B1
        r'$\beta_1$': ([0, 4], [0, 8]),   # Edges A1B1 and A1C1
        r'$\gamma_1$': ([0, 3], [0, 8]),  # Edges A1A4 and A1C1
        r'$\delta_1$': ([0, 1], [0, 3]),  # Edges A1A2 and A1A4
        
        r'$\alpha_2$': ([1, 0], [1, 5]),  # Edges A2A1 and A2B2
        r'$\beta_2$': ([1, 5], [1, 9]),   # Edges A2B2 and A2C2
        r'$\gamma_2$': ([1, 2], [1, 9]),  # Edges A2A3 and A2C2
        r'$\delta_2$': ([1, 2], [1, 0]),  # Edges A2A3 and A2A1
        
        r'$\alpha_3$': ([2, 3], [2, 6]),  # Edges A3A4 and A3B3
        r'$\beta_3$': ([2, 6], [2, 10]),  # Edges A3B3 and A3C3
        r'$\gamma_3$': ([2, 1], [2, 10]), # Edges A3A2 and A3C3
        r'$\delta_3$': ([2, 3], [2, 1]),  # Edges A3A4 and A3A2
        
        r'$\alpha_4$': ([3, 2], [3, 7]),  # Edges A4A3 and A4B4
        r'$\beta_4$': ([3, 7], [3, 11]),  # Edges A4B4 and A4C4
        r'$\gamma_4$': ([3, 0], [3, 11]), # Edges A4A1 and A4C4
        r'$\delta_4$': ([3, 0], [3, 2]),  # Edges A4A1 and A4A3
    }

    # Add angle labels at specific locations
    angle_label_list = []  # To keep track of angle labels
    for angle_name, (edge1_indices, edge2_indices) in angles_to_edges.items():
        # Get the vertices of the edges
        v1_start, v1_end = vertices_list[edge1_indices[0]], vertices_list[edge1_indices[1]]
        v2_start, v2_end = vertices_list[edge2_indices[0]], vertices_list[edge2_indices[1]]

        # Calculate vectors representing the edges
        vector1 = np.array(v1_end) - np.array(v1_start)
        vector2 = np.array(v2_end) - np.array(v2_start)

        # Calculate the angle between the two vectors
        angle_value = calculate_angle(vector1, vector2)

        # Find the common vertex for placing the label
        common_vertex = np.array(v1_start)

        # Calculate a position slightly away from the common vertex for better visibility
        offset_vector = (vector1 + vector2) / 4.0  # Average direction with some scaling
        label_position = common_vertex + offset_vector 

        # Project the label position from 3D to 2D for annotation placement
        x2d_label, y2d_label, _ = proj3d.proj_transform(
            label_position[0], label_position[1], label_position[2], ax.get_proj()
        )

        # Create annotation for the angle
        angle_label = pylab.annotate(
            f"{angle_name} = {angle_value:.2f}°", xy=(x2d_label, y2d_label), xytext=(0, 0),
            color='green', fontsize=6, textcoords='offset points',
            ha='center', va='center'
        )
        angle_label.draggable()  # Make angle labels draggable
        angle_label_list.append(angle_label)

    # Initialize variables for dihedral angles and rotation control
    phi = phi_init
    rotation_enabled = True

    # Create a rectangular patch to use as a frame for displaying dihedral angles
    frame = patches.Rectangle((0.02, 0.63), 0.12, 0.2, transform=fig.transFigure,
                            facecolor='lightgray', edgecolor='black', lw=1, alpha=0.3)
    fig.patches.append(frame)  # Add frame to the figure

    # Create text elements inside the frame for displaying dihedral angles
    phi_text = fig.text(0.03, 0.8, f'$\\varphi$ = {phi_init:.2f}$^\\circ$', fontsize=10, color='black')
    psi2_text = fig.text(0.03, 0.75, f'$\\psi_2$ = {psi2_init:.2f}$^\\circ$', fontsize=10, color='black')
    theta_text = fig.text(0.03, 0.7, f'$\\theta$ = {theta_init:.2f}$^\\circ$', fontsize=10, color='black')
    psi1_text = fig.text(0.03, 0.65, f'$\\psi_1$ = {psi1_init:.2f}$^\\circ$', fontsize=10, color='black')

    # Define function to update the plot when the slider value changes
    def update(val):
        nonlocal Vertices, vertices_list, AllEdgesToDraw, labels_list, angle_label_list, edge_label_list, phi, psi1_init, psi2_init, theta_init, DihedralAngles
        # Update dihedral angle phi from slider
        phi = phi_slider.val

        tilde_e = 1
        
        if phi != 0 and phi != 360:
            Z = 1 / tan(phi / 2)                    
            try:
                with np.errstate(invalid='raise', divide='raise'):
                    
                    W2 = (np.sqrt(2) * (2 * Z - tilde_e * np.sqrt((1 + np.sqrt(3)) * (np.sqrt(3) * Z**2 - 1) * (1 - (2 * np.sqrt(3) + 3) * Z**2)))) / ((1 + np.sqrt(3)) * (1 - 3 * Z**2))
                    
                    W1 = ((3 + 2 * np.sqrt(3)) * Z - tilde_e * np.sqrt((1 + np.sqrt(3)) * (np.sqrt(3) * Z**2 - 1) * (1 - (2 * np.sqrt(3) + 3) * Z**2))) / (1 + np.sqrt(3) + (3 + 2 * np.sqrt(3)) * Z**2)
                    
                    U = ((36 + 17 * np.sqrt(3)) * Z - tilde_e * 2 * np.sqrt(3 * (1 + np.sqrt(3)) * (np.sqrt(3) * Z**2 - 1) * (1 - (2 * np.sqrt(3) + 3) * Z**2))) / (np.sqrt(3) + 12 * (9 + 5 * np.sqrt(3)) * Z**2)


            except (FloatingPointError, ZeroDivisionError, ValueError):
                W2, W1, U = None, None, None
        else: 
            print("Dihedral Angle should be between 0 and 360 degrees.")
            return 
        if None not in [W1, W2, U] and all(np.isfinite([W1, W2, U])):
            # Angles
            psi1_angle = 2 * arctan(1 / W1)
            psi2_angle = 2 * arctan(1 / W2)
            theta_angle = 2 * arctan(1 / U)
        
    
            if psi1_angle < 0:
                psi1_angle += 360
            
            if psi2_angle < 0:
                psi2_angle += 360
        
            if theta_angle < 0:
                theta_angle += 360

            psi1 = psi1_angle
            psi2 = psi2_angle
            theta = theta_angle
    
            DihedralAngles = [phi, psi2, theta, psi1]
        
        else:
            print("Complex roots encountered.")
            return
            
            
                
        # Recompute vertices with new dihedral angles
        Vertices = convert_angles_to_vertices(Alphas, Gammas, Deltas, DihedralAngles, Edge)
        vertices_list = [list(val) for val in Vertices.values()]

        # Update AllEdgesToDraw with new vertices
        AllEdgesToDraw = [[vertices_list[i] for i in face] for face in AllFaces]

        # Update the Poly3DCollection with new vertices
        collection.set_verts(AllEdgesToDraw)

        # Update the positions of the labels and scatter points
        for i in range(num_vertices):
            x2d_list[i], y2d_list[i], _ = proj3d.proj_transform(
                vertices_list[i][0], vertices_list[i][1], vertices_list[i][2], ax.get_proj()
            )
            labels_list[i].xy = x2d_list[i], y2d_list[i]
            labels_list[i].update_positions(fig.canvas.renderer)
            labels_list[i].set_position((0, 0))  # Reset text offset
            # Update the scatter points
            scatter_plots[i]._offsets3d = (
                [vertices_list[i][0]], [vertices_list[i][1]], [vertices_list[i][2]]
            )
        
        # Update the positions of the angle labels
        for angle_label, (edge1_indices, edge2_indices) in zip(angle_label_list, angles_to_edges.values()):
            # Get the vertices of the edges
            v1_start, v1_end = vertices_list[edge1_indices[0]], vertices_list[edge1_indices[1]]
            v2_start, v2_end = vertices_list[edge2_indices[0]], vertices_list[edge2_indices[1]]

            # Calculate vectors representing the edges
            vector1 = np.array(v1_end) - np.array(v1_start)
            vector2 = np.array(v2_end) - np.array(v2_start)

            # Calculate the updated angle between the two vectors
            angle_value = calculate_angle(vector1, vector2)

            # Find the common vertex for placing the label
            common_vertex = np.array(v1_start)

            # Calculate a position slightly away from the common vertex for better visibility
            offset_vector = (vector1 + vector2) / 4  # Average direction with some scaling
            label_position = common_vertex + offset_vector  # Adjust offset distance

            # Project the label position from 3D to 2D for annotation placement
            x2d_label, y2d_label, _ = proj3d.proj_transform(
                label_position[0], label_position[1], label_position[2], ax.get_proj()
            )

            # Update the angle label text and position
            angle_label.set_text(f"{angle_label.get_text().split('=')[0]}= {angle_value:.2f}°")
            angle_label.xy = x2d_label, y2d_label
            angle_label.update_positions(fig.canvas.renderer)
            angle_label.set_position((0, 0))  # Reset text offset
        
        # Update the positions and lengths of the edge labels
        for edge, label in zip(Edges, edge_label_list):
            # Extract the edge's vertices from the vertices_list
            edge_vertices = [vertices_list[i] for i in edge]
            # Calculate the length of the edge
            edge_length = np.linalg.norm(np.array(edge_vertices[0]) - np.array(edge_vertices[1]))
            # Format the edge length to 3 significant digits
            edge_label_text = f"${edge_length:.4g}$"
            # Update the label text
            label.set_text(edge_label_text)
            # Calculate the midpoint of the edge
            midpoint = np.mean(edge_vertices, axis=0)
            x2d_mid, y2d_mid, _ = proj3d.proj_transform(midpoint[0], midpoint[1], midpoint[2], ax.get_proj())
            label.xy = x2d_mid, y2d_mid
            label.update_positions(fig.canvas.renderer)
            label.set_position((0, 0))  # Reset text offset


        # Update the text annotations for dihedral angles
        phi_text.set_text(f'$\\varphi$ = {phi:.2f}$^\circ$')
        psi2_text.set_text(f'$\\psi_2$ = {psi2:.2f}$^\circ$')
        theta_text.set_text(f'$\\theta$ = {theta:.2f}$^\circ$')
        psi1_text.set_text(f'$\\psi_1$ = {psi1:.2f}$^\circ$')

        # Redraw the canvas
        fig.canvas.draw_idle()

    def update_position(event):
        """
        Updates the position of vertex labels in 2D space based on the 3D view.

        This function is called on mouse release to adjust the position of vertex labels
        after the 3D plot has been rotated or moved.

        Parameters:
            event (Event): The Matplotlib event triggered by a mouse release.
        """
        for i in range(num_vertices):
            x2d_list[i], y2d_list[i], _ = proj3d.proj_transform(
                vertices_list[i][0], vertices_list[i][1], vertices_list[i][2], ax.get_proj()
            )
            labels_list[i].xy = x2d_list[i], y2d_list[i]
            labels_list[i].update_positions(fig.canvas.renderer)

            # Update the positions of the angle labels
        for angle_label, (edge1_indices, edge2_indices) in zip(angle_label_list, angles_to_edges.values()):
            # Get the vertices of the edges
            v1_start, v1_end = vertices_list[edge1_indices[0]], vertices_list[edge1_indices[1]]
            v2_start, v2_end = vertices_list[edge2_indices[0]], vertices_list[edge2_indices[1]]

            # Calculate vectors representing the edges
            vector1 = np.array(v1_end) - np.array(v1_start)
            vector2 = np.array(v2_end) - np.array(v2_start)

            # Calculate the updated angle between the two vectors
            angle_value = calculate_angle(vector1, vector2)

            # Find the common vertex for placing the label
            common_vertex = np.array(v1_start)

            # Calculate a position slightly away from the common vertex for better visibility
            offset_vector = (vector1 + vector2) / 4.0  # Average direction with some scaling
            label_position = common_vertex + offset_vector  # Adjust offset distance

            # Project the label position from 3D to 2D for annotation placement
            x2d_label, y2d_label, _ = proj3d.proj_transform(
                label_position[0], label_position[1], label_position[2], ax.get_proj()
            )

            # Update the angle label text and position
            angle_label.xy = x2d_label, y2d_label
            angle_label.update_positions(fig.canvas.renderer)
            
        for edge, label in zip(Edges, edge_label_list):
            # Extract the edge's vertices from the vertices_list
            edge_vertices = [vertices_list[i] for i in edge]
            # Calculate the midpoint of the edge
            midpoint = np.mean(edge_vertices, axis=0)
            x2d_mid, y2d_mid, _ = proj3d.proj_transform(midpoint[0], midpoint[1], midpoint[2], ax.get_proj())
            label.xy = x2d_mid, y2d_mid
            label.update_positions(fig.canvas.renderer)

        fig.canvas.draw_idle()

    def onkey(event):
        """
        Toggles mouse rotation for the 3D plot on and off.

        This function is triggered when the 'z' key is pressed and disables or enables
        mouse rotation of the 3D plot.

        Parameters:
            event (Event): The Matplotlib event triggered by a key press.
        """
        nonlocal rotation_enabled
        if event.key == 'z':  # Press 'z' to toggle rotation
            if rotation_enabled:
                ax.disable_mouse_rotation()  # Disable rotation
            else:
                ax.mouse_init()  # Enable rotation
            rotation_enabled = not rotation_enabled  # Toggle the flag
            plt.draw()

    # Connect event handlers
    fig.canvas.mpl_connect('key_press_event', onkey)
    fig.canvas.mpl_connect('button_release_event', update_position)

    # Create axes for the slider (positioned below the main plot)
    ax_phi = plt.axes([0.15, 0.05, 0.7, 0.02])  # Slider for phi

    # Create the slider to control the dihedral angle phi
    phi_slider = Slider(
        ax=ax_phi,
        label=r'Angle $\varphi$ (degrees)',
        valmin=0.0,
        valmax=360.0,
        valinit=phi,
        valstep=0.01,
        valfmt='%1.2f',
        color='royalblue'
    )

    # Create axes for the buttons
    ax_button_minus = plt.axes([0.15, 0.01, 0.03, 0.02])  # Button to decrement phi
    ax_button_plus = plt.axes([0.81, 0.01, 0.03, 0.02])   # Button to increment phi

    # Create the buttons
    button_minus = Button(ax_button_minus, '-', color='lightgray', hovercolor='gray')
    button_plus = Button(ax_button_plus, '+', color='lightgray', hovercolor='gray')
    
    # Create axes for the buttons
    ax_button_vertices = plt.axes([0.02, 0.25, 0.12, 0.03])  # Button to toggle vertices visibility
    ax_button_edges = plt.axes([0.02, 0.2, 0.12, 0.03])     # Button to toggle edges visibility

    # Create the buttons
    button_vertices = Button(ax_button_vertices, 'Toggle Vertices', color='lightgray', hovercolor='gray')
    button_edges = Button(ax_button_edges, 'Toggle Edges', color='lightgray', hovercolor='gray')
    
    # Create axes for the buttons to toggle visibility of different angle labels
    ax_button_alphas = plt.axes([0.02, 0.5, 0.12, 0.03])  # Button to toggle alpha labels visibility
    ax_button_betas = plt.axes([0.02, 0.45, 0.12, 0.03])   # Button to toggle beta labels visibility
    ax_button_gammas = plt.axes([0.02, 0.4, 0.12, 0.03])   # Button to toggle gamma labels visibility
    ax_button_deltas = plt.axes([0.02, 0.35, 0.12, 0.03])  # Button to toggle delta labels visibility
    ax_button_all_angles = plt.axes([0.02, 0.3, 0.12, 0.03])  # Button to toggle all angle labels visibility

    # Create the buttons
    button_alphas = Button(ax_button_alphas, 'Toggle Alphas', color='lightgray', hovercolor='gray')
    button_betas = Button(ax_button_betas, 'Toggle Betas', color='lightgray', hovercolor='gray')
    button_gammas = Button(ax_button_gammas, 'Toggle Gammas', color='lightgray', hovercolor='gray')
    button_deltas = Button(ax_button_deltas, 'Toggle Deltas', color='lightgray', hovercolor='gray')
    button_all_angles = Button(ax_button_all_angles, 'Toggle All Angles', color='lightgray', hovercolor='gray')
    
    
    # -------------------- ZOOM CONTROLS --------------------
    # Store initial limits for zoom calculation
    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()
    initial_zlim = ax.get_zlim()
    zoom_level = 100.0  # Start at 100% zoom

    # Create axes for zoom controls below the 'Toggle Edges' button
    ax_zoom_out = plt.axes([0.02, 0.15, 0.05, 0.03])
    ax_zoom_in = plt.axes([0.09, 0.15, 0.05, 0.03])
    # Create an invisible axes for text to align it properly
    ax_zoom_text = plt.axes([0.02, 0.1, 0.12, 0.03], frameon=False)
    ax_zoom_text.set_xticks([])
    ax_zoom_text.set_yticks([])

    # Create zoom buttons
    button_zoom_out = Button(ax_zoom_out, '-', color='lightgray', hovercolor='gray')
    button_zoom_in = Button(ax_zoom_in, '+', color='lightgray', hovercolor='gray')

    # Create zoom percentage text display
    zoom_text = ax_zoom_text.text(0.5, 0.5, f'Zoom: {zoom_level:.0f}%',
                                  ha='center', va='center', fontsize=10)

    # Define zoom callback functions
    def zoom(factor):
        """Zooms the 3D plot by a given factor."""
        nonlocal zoom_level
        zoom_level *= factor

        # Get current camera center
        x_center = np.mean(ax.get_xlim())
        y_center = np.mean(ax.get_ylim())
        z_center = np.mean(ax.get_zlim())

        # Get initial range
        x_range_initial = initial_xlim[1] - initial_xlim[0]
        y_range_initial = initial_ylim[1] - initial_ylim[0]
        z_range_initial = initial_zlim[1] - initial_zlim[0]

        # Calculate new range based on zoom level
        scale_factor = 100.0 / zoom_level
        new_x_range = x_range_initial * scale_factor
        new_y_range = y_range_initial * scale_factor
        new_z_range = z_range_initial * scale_factor
        
        # Set new limits centered on the current view
        ax.set_xlim([x_center - new_x_range / 2, x_center + new_x_range / 2])
        ax.set_ylim([y_center - new_y_range / 2, y_center + new_y_range / 2])
        ax.set_zlim([z_center - new_z_range / 2, z_center + new_z_range / 2])
        
        # Update text display
        zoom_text.set_text(f'Zoom: {zoom_level:.0f}%')
        fig.canvas.draw_idle()

    def zoom_in(event):
        """Callback to zoom in."""
        zoom(1.1)  # Zoom in by 10%

    def zoom_out(event):
        """Callback to zoom out."""
        zoom(1 / 1.1)  # Zoom out by 10%

    # Connect callbacks to zoom buttons
    button_zoom_in.on_clicked(zoom_in)
    button_zoom_out.on_clicked(zoom_out)


    # Define the callback functions for the buttons
    def decrement_phi(event):
        new_phi = max(phi_slider.val - 0.01, phi_slider.valmin)
        phi_slider.set_val(new_phi)

    def increment_phi(event):
        new_phi = min(phi_slider.val + 0.01, phi_slider.valmax)
        phi_slider.set_val(new_phi)
        
        
    # Toggle visibility of vertex scatter plots
    vertices_visible = [True]  # Use a list to modify within the callback function

    def toggle_vertices(event):
        vertices_visible[0] = not vertices_visible[0]
        for scatter_plot in scatter_plots:
            scatter_plot.set_visible(vertices_visible[0])
        for label in labels_list:
            label.set_visible(vertices_visible[0])
        # Redraw the canvas to reflect changes
        fig.canvas.draw_idle()

    # Toggle visibility of edge labels
    edges_visible = [True]  # Use a list to modify within the callback function

    def toggle_edges(event):
        edges_visible[0] = not edges_visible[0]
        for label in edge_label_list:
            label.set_visible(edges_visible[0])
        # Redraw the canvas to reflect changes
        fig.canvas.draw_idle()
        

    # Initialize visibility states for angle labels
    alphas_visible = [True]
    betas_visible = [True]
    gammas_visible = [True]
    deltas_visible = [True]

    # Define callback functions for each button to toggle angle labels visibility
    def toggle_alphas(event):
        alphas_visible[0] = not alphas_visible[0]
        for i, angle_name in enumerate(angles_to_edges.keys()):
            if r'$\alpha' in angle_name:
                angle_label_list[i].set_visible(alphas_visible[0])
        fig.canvas.draw_idle()

    def toggle_betas(event):
        betas_visible[0] = not betas_visible[0]
        for i, angle_name in enumerate(angles_to_edges.keys()):
            if r'$\beta' in angle_name:
                angle_label_list[i].set_visible(betas_visible[0])
        fig.canvas.draw_idle()

    def toggle_gammas(event):
        gammas_visible[0] = not gammas_visible[0]
        for i, angle_name in enumerate(angles_to_edges.keys()):
            if r'$\gamma' in angle_name:
                angle_label_list[i].set_visible(gammas_visible[0])
        fig.canvas.draw_idle()

    def toggle_deltas(event):
        deltas_visible[0] = not deltas_visible[0]
        for i, angle_name in enumerate(angles_to_edges.keys()):
            if r'$\delta' in angle_name:
                angle_label_list[i].set_visible(deltas_visible[0])
        fig.canvas.draw_idle()

    def toggle_all_angles(event):
        # Toggle all angles by toggling each set individually
        all_visible = not (alphas_visible[0] or betas_visible[0] or gammas_visible[0] or deltas_visible[0])

        # Set visibility for all to the same state (toggle all at once)
        alphas_visible[0] = betas_visible[0] = gammas_visible[0] = deltas_visible[0] = all_visible

        for label in angle_label_list:
            label.set_visible(all_visible)

        fig.canvas.draw_idle()


    # Connect the callbacks to the buttons
    button_minus.on_clicked(decrement_phi)
    button_plus.on_clicked(increment_phi)
    
    # Connect the buttons to the callback functions
    button_vertices.on_clicked(toggle_vertices)
    button_edges.on_clicked(toggle_edges)
    
    # Connect the buttons to the callback functions
    button_alphas.on_clicked(toggle_alphas)
    button_betas.on_clicked(toggle_betas)
    button_gammas.on_clicked(toggle_gammas)
    button_deltas.on_clicked(toggle_deltas)
    button_all_angles.on_clicked(toggle_all_angles)

    # Connect the update function to the slider
    phi_slider.on_changed(update)

    # Display the plot
    plt.show()

def create_obj_file_of_vertices_and_faces(vertices, i=None):
    """
    Creates an OBJ file with the given vertices and faces for 3D modeling.

    This function generates a 3D object file in OBJ format by converting 
    the provided vertex coordinates and face definitions into the required format.

    Parameters:
        vertices (dict): A dictionary where each key is a vertex label (string),
                         and each value is a tuple of coordinates (x, y, z).

    Output:
        A file named "VerticesFaces.obj" containing the vertices and faces in OBJ format.
    """
    # Initialize the vertex text in OBJ format
    vertex_text = ""
    for val in vertices.values():
        # Start each line with 'v' indicating a vertex in OBJ format
        vertex_text += "v"
        for coordinate in val:
            # Append each coordinate to the line with appropriate formatting
            vertex_text += f" {coordinate}"
        vertex_text += "\n"  # Move to the next line for the next vertex

    # Define the faces (polygon surfaces) for the object
    face_text = (
        "f 1 2 3 4\n"     # Face connecting vertices 1, 2, 3, 4
        "f 1 5 6 2\n"     # Face connecting vertices 1, 5, 6, 2
        "f 2 6 10\n"      # Face connecting vertices 2, 6, 10
        "f 2 10 11 3\n"   # Face connecting vertices 2, 10, 11, 3
        "f 3 11 7\n"      # Face connecting vertices 3, 11, 7
        "f 3 7 8 4\n"     # Face connecting vertices 3, 7, 8, 4
        "f 4 8 12\n"      # Face connecting vertices 4, 8, 12
        "f 4 12 9 1\n"    # Face connecting vertices 4, 12, 9, 1
        "f 1 9 5\n"       # Face connecting vertices 1, 9, 5
    )

    # Combine the vertex and face information into a single OBJ string
    obj_text = vertex_text + face_text

    # Write the text to an OBJ file
    if i is None:
        nameT = 0
    else:
        nameT = i
    try:
        with open(f"VerticesFaces_{nameT}.obj", "w") as obj_file:
            obj_file.write(obj_text)
        print(f"OBJ file 'VerticesFaces_{nameT}.obj' created successfully.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def for_geogebra(vertices):
    """
    Creates a text file formatted for use with GeoGebra, including vertices 
    and polygon definitions to form the faces of a 3D shape.

    Parameters:
        vertices (dict): A dictionary where each key is a vertex label (string),
                         and each value is a tuple of coordinates (x, y, z).

    Output:
        A file named "forGeoGebra.txt" containing the vertices and polygons in GeoGebra format.
    """
    # Generate the vertex declarations for GeoGebra
    vertices_text = ""
    for label, coord in vertices.items():
        # Format the vertex declaration for GeoGebra
        vertices_text += f"{label} = ({coord[0]}, {coord[1]}, {coord[2]})\n"

    # Define the polygons (faces) for GeoGebra using the defined vertices
    polygon_text = (
        "\nPolygon($A_1$, $A_2$, $A_3$, $A_4$)\n"    # Polygon face A1A2A3A4
        "Polygon($A_1$, $B_1$, $B_2$, $A_2$)\n"      # Polygon face A1B1B2A2
        "Polygon($A_2$, $B_2$, $C_2$)\n"             # Polygon face A2B2C2
        "Polygon($A_2$, $C_2$, $C_3$, $A_3$)\n"      # Polygon face A2C2C3A3
        "Polygon($A_3$, $C_3$, $B_3$)\n"             # Polygon face A3C3B3
        "Polygon($A_3$, $B_3$, $B_4$, $A_4$)\n"      # Polygon face A3B3B4A4
        "Polygon($A_4$, $B_4$, $C_4$)\n"             # Polygon face A4B4C4
        "Polygon($A_4$, $C_4$, $C_1$, $A_1$)\n"      # Polygon face A4C4C1A1
        "Polygon($A_1$, $C_1$, $B_1$)\n"             # Polygon face A1C1B1
    )

    # Combine the vertex declarations and polygon definitions
    geogebra_text = vertices_text + polygon_text

    # Write the text to a file
    try:
        with open("forGeoGebra.txt", "w") as geogebra_file:
            geogebra_file.write(geogebra_text)
        print("GeoGebra file 'forGeoGebra.txt' created successfully.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")


def main():
    """
    Main function to set up the parameters, compute the vertices from angles, 
    and generate outputs for visualization and file creation.
    
    The function initializes the angles, edge lengths, and dihedral angles, 
    then calls necessary functions to:
    1. Convert the angles into 3D vertices.
    2. Export the vertices for GeoGebra.
    3. Create an .obj file for 3D visualization.
    4. Visualize the 3D shape in a Matplotlib window.
    """

    # -------------------- SET-UP: Define angles and edge lengths --------------------
    # Uncomment the following lines if you wish to read input values from a file
    # file_path = 'optimal_angles.txt'  # Replace with the actual file path
    # values = read_values_from_file(file_path)
    # Alphas = values['Alphas']
    # Betas = values['Betas']
    # Gammas = values['Gammas']
    # Deltas = values['Deltas']
    # global DihedralAngles_Init
    # DihedralAngles_Init = values['DihedralAngles']    
    
    global r1, r2, r3, r4
    global s1, s2, s3, s4
    global f1, f2, f3, f4
    global M
    global Epsilons
    global Es
    global TildeEpsilons
    
    # Alphas = [np.degrees(np.arccos(1 / np.sqrt(5))), np.degrees(np.arccos(1 / (4 * np.sqrt(11)))), np.degrees(np.arccos(1 / (4 * np.sqrt(11)))), np.degrees(np.arccos(1 / np.sqrt(5)))]
    # Betas = [np.degrees(np.arccos(7 / (5 * np.sqrt(2)))), np.degrees(np.arccos(7 * np.sqrt(7) / (4 * np.sqrt(22)))), np.degrees(np.arccos(7 * np.sqrt(7) / (4 * np.sqrt(22)))), np.degrees(np.arccos(7 / (5 * np.sqrt(2))))]
    # Gammas = [np.degrees(np.arccos(-1 / np.sqrt(10))), np.degrees(np.arccos(-1 / (2 * np.sqrt(2)))), np.degrees(np.arccos(1 / (2 * np.sqrt(2)))), np.degrees(np.arccos(1 / np.sqrt(10)))]
    # Deltas = [90, 90, 90, 90]
    
    Alphas = [105, 90, 90, 105]
    Betas = [15, 120, 60, 15]
    Gammas = [120, 15, 165, 120]
    Deltas = [90, 105, 75, 90]
    
    # print(Alphas)
    # print(Betas)
    # print(Gammas)
    # print(Deltas)
    
    # Define functions
    # Define constants
    sqrt3 = np.sqrt(3)
    ttilde_e = 1
    def WW1(Z):
        return ((3 + 2 * sqrt3) * Z - ttilde_e * np.sqrt((1 + sqrt3) * (sqrt3 * Z**2 - 1) * (1 - (2 * sqrt3 + 3) * Z**2))) / \
            (1 + sqrt3 + (3 + 2 * sqrt3) * Z**2)

    def WW2(Z):
        return (np.sqrt(2) * (2 * Z - ttilde_e * np.sqrt((1 + sqrt3) * (sqrt3 * Z**2 - 1) * (1 - (2 * sqrt3 + 3) * Z**2)))) / \
            ((1 + sqrt3) * (1 - 3 * Z**2))

    def UU(Z):
        return ((36 + 17 * sqrt3) * Z - ttilde_e * 2 * np.sqrt(3 * (1 + sqrt3) * (sqrt3 * Z**2 - 1) * (1 - (2 * sqrt3 + 3) * Z**2))) / \
            (sqrt3 + 12 * (9 + 5 * sqrt3) * Z**2)
    
    ZZ = 3/4
    WW1_vals = WW1(ZZ)
    WW2_vals = WW2(ZZ)
    UU_vals = UU(ZZ)
    
    DihedralAngles = [2 * np.degrees(np.arctan(1/ZZ)), 2 * np.degrees(np.arctan(1/WW2_vals)), 2 * np.degrees(np.arctan(1/UU_vals)), 2 * np.degrees(np.arctan(1/WW1_vals))]
    
    # DihedralAngles = [2 * np.degrees(np.arctan(np.sqrt(np.sqrt(3)))), 360 + 2 * np.degrees(np.arctan(-np.sqrt(np.sqrt(3) / 2))), 2 * np.degrees(np.arctan(13 * np.sqrt(3 * np.sqrt(3)) / (24 - 7 * np.sqrt(3)))), 2 * np.degrees(np.arctan(np.sqrt(np.sqrt(3))))]


    Edge = 4.0  # Length of the edge A1A2

    # Initialize the global dihedral angles
    global DihedralAngles_Init
    
    DihedralAngles_Init = DihedralAngles
    
    # -------------------- RESULT: Compute vertices and generate outputs --------------------
    # Convert angles and edge length to vertices
    # Vertices = convert_angles_to_vertices(Alphas, Gammas, Deltas, DihedralAngles_Init, Edge)

    # Generate a file for GeoGebra containing the vertices and polygons
    # for_geogebra(Vertices)

    # Create an .obj file for use in 3D modeling software
                    
    # phi = 105.37
    # tilde_e = 1
    # i = 0
    # while phi <= 115.37:
    #     i+=1
    #     if phi != 0 and phi != 360:
    #         Z = 1 / tan(phi / 2)                    
    #         try:
    #             with np.errstate(invalid='raise', divide='raise'):
    #                 W2 = (6 * np.sqrt(10) * Z - 2 * tilde_e * np.sqrt(3 * (2 - 5 * Z**2) * (5 * Z**2 - 1))) / (3 + 15 * Z**2)

    #                 W1 = (15 * np.sqrt(3) * Z - 3 * tilde_e * np.sqrt(2 * (2 - 5 * Z**2) * (5 * Z**2 - 1))) / (1 + 10 * Z**2)
                    
    #                 U = 1 / Z

    #         except (FloatingPointError, ZeroDivisionError, ValueError):
    #             W2, W1, U = None, None, None
    #     else: 
    #         print("Dihedral Angle should be between 0 and 360 degrees.")
    #         return 
    #     if None not in [W1, W2, U] and all(np.isfinite([W1, W2, U])):
    #         # Angles
    #         psi1_angle = 2 * arctan(1 / W1)
    #         psi2_angle = 2 * arctan(1 / W2)
    #         theta_angle = 2 * arctan(1 / U)
        

    #         if psi1_angle < 0:
    #             psi1_angle += 360
            
    #         if psi2_angle < 0:
    #             psi2_angle += 360
        
    #         if theta_angle < 0:
    #             theta_angle += 360

    #         psi1 = psi1_angle
    #         psi2 = psi2_angle
    #         theta = theta_angle

    #         DihedralAngles = [phi, psi2, theta, psi1]
            
    #         Vertices = convert_angles_to_vertices(Alphas, Gammas, Deltas, DihedralAngles, Edge)
    #         create_obj_file_of_vertices_and_faces(Vertices, i)
    #         phi = phi + (115.37 - 105.37) / 100
    #     else:
    #         print("Complex roots encountered.")
    #         return
    # create_obj_file_of_vertices_and_faces(Vertices)

    # Visualize the 3D vertices and faces in a Matplotlib plot
    visualize_here(Alphas, Betas, Gammas, Deltas, Edge)

if __name__ == "__main__":
    main()
    
