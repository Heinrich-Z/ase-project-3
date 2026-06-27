import numpy as np
import math

# define material constant
E_11 = 126780.0  # TODO: check for 0.9 knockdown factor
E_22 = 9678.0
G_12 = 6213.6
nu_12 = 0.32
nu_21 = nu_12 * E_22 / E_11

R_lt = 3050
R_lc = 1500
R_pt = 300
R_pc = 50
R_pl = 100
p = 0.25

l = 600
w = 200
AR = l / w  # aspect ratio alpha

sigma_uc = 650

# get the full stacking sequence
def get_stack_seq(stack_seq_s):
    stack_seq = stack_seq_s + stack_seq_s[::-1]
    return stack_seq

# get the unified Q matrix of this material
def get_Q_matrix():
    Q_11 = E_11 / (1 - nu_12 * nu_21)
    Q_22 = E_22 / (1 - nu_12 * nu_21)
    Q_12 = nu_12 * E_22 / (1 - nu_12 * nu_21)
    Q_66 = G_12

    Q_mat = np.matrix([
        [Q_11, Q_12, 0.0],
        [Q_12, Q_22, 0.0],
        [0.0, 0.0, Q_66]
    ])
    return Q_mat

# get the strain transformation matrix for a ply
def strain_transform_mat(theta):
    # transform strain from problem to material COS
    theta = np.radians(theta)
    m = np.cos(theta)
    n = np.sin(theta)

    T_epsilon = np.matrix([
        [m**2, n**2, n * m],
        [n**2, m**2, -n * m],
        [-2 * n * m, 2 * n * m, m**2 - n**2]
    ])
    return T_epsilon

# get the stress transformation matrix for a ply
def stress_transform_mat(theta):
    # transform stress from problem to material COS
    theta = np.radians(theta)
    m = np.cos(theta)
    n = np.sin(theta)

    T_sigma = np.matrix([
        [m**2, n**2, 2 * n * m],
        [n**2, m**2, -2 * n * m],
        [-n * m, n * m, m**2 - n**2]
    ])
    return T_sigma

# get the Q bar matrix
def get_Q_bar_mat(Q, theta):
    T_sigma = stress_transform_mat(theta)
    T_epsilon = strain_transform_mat(theta)
    Q_bar = np.linalg.inv(T_sigma) @ Q @ T_epsilon
    return Q_bar

# get the ABD matrix of a composite laminate
def get_ABD_mat(ply_stack):
    """
    ply_stack: {
        Q_bar_{k}: numpy matrix shape 3 x 3,
        z_bottom: float
        z_top: float
    }
    """
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    for i in range(len(ply_stack['Q_bar'])):
        Q_bar = ply_stack['Q_bar'][i]
        z_k0 = ply_stack['z_k0'][i] # z_k
        z_k1 = ply_stack['z_k1'][i] # z_k+1

        A += Q_bar * (z_k1 - z_k0)
        B += (1/2) * Q_bar * (z_k1**2 - z_k0**2)
        D += (1/3) * Q_bar * (z_k1**3 - z_k0**3)

    A = np.matrix(A)
    B = np.matrix(B)
    D = np.matrix(D)

    ABD = np.bmat([
        [A, B],
        [B, D]
    ])

    return ABD

# Puck failure mode analysis
def puck_failure_analysis(sigma_1, sigma_2, tau_21):
    # 1. Fiber Fracture (FF)
    if sigma_1 > 0:
        RF_FF = R_lt / sigma_1
    else:
        RF_FF = R_lc / abs(sigma_1)

    # 2. Inter Fiber Fracture (IFF)
    R_pp_A = R_pc / (2 * (1 + p)) # Action-plane fracture resistance
    tau_21c = R_pl * math.sqrt(1 + 2 * p) # B-C transition point

    if sigma_2 >= 0:
        # Mode A
        mode = 'A'
        term1 = (tau_21 / R_pl)**2
        term2 = ((1.0 - p * (R_pt / R_pl))**2) * ((sigma_2 / R_pt)**2)
        term3 = p * (sigma_2 / R_pl)
        fE_IFF = math.sqrt(term1 + term2) + term3
        theta_fp = 0.0

    else: # sigma_2 < 0
        ratio_stress = abs(sigma_2 / tau_21)
        ratio_resistance = R_pp_A / abs(tau_21c)

        if ratio_stress <= ratio_resistance:
            # Mode B
            mode = 'B'
            term1 = 1.0 / R_pl
            term2 = math.sqrt(tau_21**2 + (p * sigma_2)**2)
            term3 = p * sigma_2
            fE_IFF = term1 * (term2 + term3)
            theta_fp = 0.0

        else:
            # Mode C
            mode = 'C'
            term1 = (tau_21 / (2.0 * (1.0 + p) * R_pl))**2
            term2 = (sigma_2 / R_pc)**2
            term3 = R_pc / abs(sigma_2)
            fE_IFF = (term1 + term2) * term3

            # Fracture plane angle calculation
            angle_term_1 = (tau_21 / sigma_2)**2 * (R_pp_A / R_pl)**2 + 1.0
            angle_term_2 = 2.0 * (1.0 + p)
            cos_theta_fp = math.sqrt(angle_term_1 / angle_term_2)

            cos_theta_fp = max(-1.0, min(1.0, cos_theta_fp))
            theta_fp = math.degrees(math.acos(cos_theta_fp))

    return RF_FF, mode, fE_IFF, theta_fp

# -------- Buckling Analysis Package -------- #
def panel_buckling(ABD_mat, t, beta):
    # 1. Buckling for biaxial loading
    min_sigma_crit = math.inf
    m_val = 0
    n_val = 0
    for m in range(1, 11):
        for n in range(1, 11):
            term1 = math.pi**2 / (t * w**2)
            term2 = 1 / ((m/AR)**2 + beta * n**2)
            term3 = ABD_mat[3][3] * (m / AR) ** 4 + 2 * (ABD_mat[3][4] + ABD_mat[5][5]) * (m * n / AR) ** 2 + ABD_mat[4][4] * n ** 4
            sigma_cr = term1 * term2 * term3
            if min_sigma_crit > sigma_cr >= 0:
                min_sigma_crit = sigma_cr
                m_val = m
                n_val = n

    # 2. Buckling for shear loading
    delta = math.sqrt(ABD_mat[3][3] * ABD_mat[4][4]) / (ABD_mat[3][4] + 2 * ABD_mat[5][5])
    if delta >= 1:
        tau_crit = 4 / (t * w**2) * ((ABD_mat[3][3] * ABD_mat[4][4] ** 3) ** 0.25 * (8.12 + 5.05 / delta))
    else: # delta < 1
        tau_crit = 4 / (t * w**2) * (math.sqrt(ABD_mat[4][4] * (ABD_mat[3][4] + 2 * ABD_mat[5][5])) * (11.7 + 0.532 * delta + 0.938 * delta ** 2))

    return min_sigma_crit, m_val, n_val, tau_crit

def Iyy_beam(b, h):
    return b * h**3 / 12

def slenderness(Iyy, area, length, c):
    radius_of_gyration = np.sqrt(Iyy / area)
    print("Radius of Gyration:", radius_of_gyration)
    lamda = length * c / radius_of_gyration
    return lamda

def crippling_stress(bm, t, case):
    if case == 'OEF':  # one edge free (OEF)
        sigma_crip = 1.63 * sigma_uc / (bm / t)**0.717

    elif case == 'NOF':  # no edge free (NOF)
        sigma_crip = 11.0 * sigma_uc / (bm / t)**1.124

    else: # segment does not cripple
        sigma_crip = sigma_uc

    return sigma_crip

def Euler_Johnson(sigma_uc, sigma_crip, lamda, E_comb):
    sigma_cutoff = float(min(sigma_uc, sigma_crip))
    lamda_crit = np.sqrt(2 * E_comb * np.pi**2 / sigma_cutoff)
    print("lamda_crit: ", lamda_crit)
    if lamda < lamda_crit:
        # proceed with E-J
        sigma_ej = sigma_cutoff - sigma_cutoff**2 * lamda**2 / (4 * np.pi**2 * E_comb)
    else:
        # proceed with Euler
        sigma_ej = (np.pi ** 2 * E_comb) / lamda**2

    return sigma_cutoff, sigma_ej

def get_mid_line(*args):
    """
    :param args: [sec_name, ]
    :return:
    """
    return

def t_stringer_buckling_analysis(pitch, skin_t, web_t, web_height, flange_width, flange_t, *args):
    """
    pitch is the width of a panel element
    use mid-line value for computing EIy and Iyy
    args: ABD_flange, ABD_web, ABD_skin
    variable sequence: flange, web, left_skin, right_skin
    """
    # 1. Second moment of area
    # compute the mid-line length
    bm_flange = flange_width
    hm_web = web_height - flange_t / 2
    half_pitch = pitch / 2

    # compute the elastic center of each element
    zi_left_skin = -skin_t / 2
    zi_right_skin = -skin_t / 2
    zi_flange = flange_t / 2
    zi_web = flange_t / 2 + hm_web / 2

    #
    area_left_skin = skin_t * half_pitch
    area_right_skin = skin_t * half_pitch
    area_flange = bm_flange * flange_t
    area_web = web_t * hm_web

    zi_list = np.array([zi_flange, zi_web, zi_left_skin, zi_right_skin])
    area_list = np.array([area_flange, area_web, area_left_skin, area_right_skin])
    area = area_list.sum()

    z_ec = np.dot(zi_list, area_list) / area
    steiner = (zi_list - z_ec)**2 * area_list
    sec_area = np.array([Iyy_beam(bm_flange, flange_t), Iyy_beam(web_t, hm_web), Iyy_beam(half_pitch, skin_t), Iyy_beam(half_pitch, skin_t)])
    Iyy = steiner.sum() + sec_area.sum()

    # 2. Composite engineering constant
    # skin: bending, constrained
    ABD_flange = args[0]
    E_flange_b = 12 * ABD_flange[3][3] / skin_t**3
    E_flange_x = ABD_flange[0][0] / skin_t

    # flange: bending, constrained
    ABD_web = args[1]
    E_web_b = 12 * ABD_web[3][3] / web_t
    E_web_x = ABD_web[0][0] / web_t

    # web: axial, free
    ABD_skin = args[2]
    ABD_inv_skin = np.linalg.inv(ABD_skin)
    E_skin_b = 1 / (ABD_inv_skin[0][0] * skin_t)
    E_skin_x = 1 / (ABD_inv_skin[0][0] * skin_t)

    eng_const = {'flange': [E_flange_b, E_flange_x], 'web': [E_web_b, E_web_x], 'skin': [E_skin_x, E_skin_b]}
    
    # 3. Combined Bending stiffness
    comb_stiff_flange = eng_const['flange'][0] * sec_area[0] + eng_const['flange'][1] * steiner[0]
    comb_stiff_web = eng_const['web'][0] * sec_area[1] + eng_const['flange'][1] * steiner[1]
    com_stiff_left_skin = eng_const['skin'][0] * sec_area[2] + eng_const['skin'][1] * steiner[2]
    com_stiff_right_skin = eng_const['skin'][0] * sec_area[3] + eng_const['skin'][1] * steiner[3]

    comb_stiff_list = np.array([comb_stiff_flange, comb_stiff_web, com_stiff_left_skin, com_stiff_right_skin])
    comb_stiff = comb_stiff_list.sum()

    return Iyy, area, z_ec, eng_const, comb_stiff

def t_stringer_crip_analysis(flange_width, flange_t, web_width, web_height):
    # ai_j represents the geometric value
    a1_1 = flange_width / 2
    a1_2 = web_height
    t1 = flange_t
    t2 = web_width

    # bi_j represents the mid-line value
    b1_1 = a1_1
    b1_2 = a1_2 + t1 / 2

    sigma_crip_1 = crippling_stress(b1_1, t1, 'DNC') # flange (attached to skins)
    sigma_crip_2 = crippling_stress(b1_2, t2, 'OEF') # web

    # to compute the combined crippling stress the geometric dimensions are used
    sigma_crip_avg = (2 * sigma_crip_1 * a1_1 * t1 + sigma_crip_2 * a1_2 * t2) / (2 * a1_1 * t1 + a1_2 * t2)
    return sigma_crip_avg

def omega_stringer_Iyy(pitch, skin_t, web_t, web_height, flange_width, flange_t, *args):
    """
    :param pitch:
    :param skin_t:
    :param web_t:
    :param web_height:
    :param flange_width:
    :param flange_t:
    :param args:
    :return:
    """
    # Omega-stringer second moment of area calculation
    zi_left_skin = -left_skin_t / 2
    zi_right_skin = -right_skin_t / 2
    zi_top_flange = t / 2 # left and right
    zi_web = web_height / 2 # left and right
    zi_lower_flange = web_height - t / 2

    area_left_skin = left_pitch * left_skin_t
    area_right_skin = right_pitch * right_skin_t
    area_top_flange = 2 * top_flange_width * t # left and right
    area_web = 2 * t * web_height # left and right
    area_lower_flange = (lower_flange_width - 2 * t) * t

    zi_list = np.array([zi_left_skin, zi_right_skin, zi_top_flange, zi_web, zi_lower_flange])
    area_list = np.array([area_left_skin, area_right_skin, area_top_flange, area_web, area_lower_flange])
    area = area_list.sum()

    z_ec = np.dot(zi_list, area_list) / area
    steiner = (zi_list - z_ec)**2 * area_list
    Iyy = steiner.sum() + 2 * Iyy_beam(top_flange_width, t) + 2 * Iyy_beam(t, web_height) + Iyy_beam(left_pitch, left_skin_t) + Iyy_beam(right_pitch, right_skin_t) + Iyy_beam(lower_flange_width - 2 * t, t)

    print("Skin Area: ", area_list[0:2].sum(), "\nStringer Area: ", area_list[2:].sum())
    return Iyy, area, z_ec