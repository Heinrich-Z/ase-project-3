from math import inf

import numpy as np
import pandas as pd
import math

# define material constant
E_11 = 126780.0 # TODO: check for 0.9 knockdown factor
E_22 = 9678.0
G_12 = 6213.6
nu_12 = 0 # TODO: to be modified
nu_21 = nu_12 * E_22 / E_11

R_lt = 3050
R_lc = 1500
R_pt = 300
R_pc = 50
R_pl = 100
p = 0.25

a = 600
b = 200
AR = a / b


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

def get_Q_bar_mat(Q, theta):
    T_sigma = stress_transform_mat(theta)
    T_epsilon = strain_transform_mat(theta)
    Q_bar = np.linalg.inv(T_sigma) @ Q @ T_epsilon
    return Q_bar

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

    for ply in ply_stack:
        Q_bar = ply['Q_bar']
        z_k0 = ply['z_bottom']
        z_k1 = ply['z_top']

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

    return ABD, A, B, D

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

def panel_buckling(D, t, beta):
    # 1. Buckling for biaxial loading
    min_sigma_crit = math.inf
    m_val = 0
    n_val = 0
    for m in range(1, 11):
        for n in range(1, 11):
            term1 = math.pi**2 / (t * b**2)
            term2 = 1
            term3 = D[0][0] * (m / AR)**4 + 2 * (D[0][1] + D[2][2]) * (m * n / AR)**2 + D[1][1] * n**4
            sigma_cr = term1 * term2 * term3
            if min_sigma_crit > sigma_cr >= 0:
                min_val = sigma_cr
                m_val = m
                n_val = n

    # 2. Buckling for shear loading
    delta = math.sqrt(D[0][0] * D[1][1]) / (D[0][1] + 2 * D[2][2])
    if delta >= 1:
        tau_cr = 4 / (t * b**2) * ((D[0][0] * D[1][1]**3)**0.25 * (8.12 + 5.05 / delta))


def Iyy_beam(b, h):
    return b * h**3 / 12

def slenderness(Iyy, area):
    radius_of_gyration = np.sqrt(Iyy / area)
    print("Radius of Gyration:", radius_of_gyration)
    lamda = length * c / radius_of_gyration
    return lamda

def crippling_stress(bi, ti, num_support, Ec, sigma_p, crip):
    Ki = 0.41 if num_support == 'single' else 3.6
    xi = bi / ti * np.sqrt(sigma_p/(Ki * Ec))

    if xi > 1.633:
        alpha = 0.69/(xi ** 0.75)
    elif xi <= 1.633 and xi > 1.095:
        alpha = 0.78/xi
    elif xi <= 1.095 and xi >= 0.4:
        alpha = 1.4 - 0.628 * xi
    elif xi < 0.4:
        # in this case the part does not cripple and sigma_crip = sigma_p
        alpha = 1
    else:
        print("Error with xi calculation")

    if not crip:
        # if the part does not cripple, then sigma_crip = sigma_p
        alpha = 1

    return alpha * sigma_p

def Euler_Johnson(sigma_p, sigma_crip, lamda):
    sigma_cutoff = min(sigma_p, sigma_crip)
    lamda_crit = np.sqrt(2 * Ec * np.pi**2 / sigma_cutoff)
    print("lamda_crit: ", lamda_crit)
    if lamda < lamda_crit:
        # proceed with E-J
        sigma_ej = sigma_cutoff - sigma_cutoff**2 * lamda**2 / (4 * np.pi**2 * Ec)
    else:
        # proceed with Euler
        sigma_ej = (np.pi ** 2 * Ec) / lamda**2

    return sigma_ej

def t_stringer_Iyy(left_pitch, right_pitch, left_skin_t, right_skin_t,
                   web_width, web_height, flange_width, flange_t):

    # T-stringer second moment of area calculation
    zi_left_skin = -left_skin_t / 2
    zi_right_skin = -right_skin_t / 2
    zi_flange = flange_t / 2
    zi_web = (web_height + flange_t) / 2

    area_left_skin = left_pitch * left_skin_t
    area_right_skin = right_pitch * right_skin_t
    area_flange = flange_width * flange_t
    area_web = web_width * (web_height - flange_t)

    zi_list = np.array([zi_left_skin, zi_right_skin, zi_flange, zi_web])
    area_list = np.array([area_left_skin, area_right_skin, area_flange, area_web])
    area = area_list.sum()

    z_ec = np.dot(zi_list, area_list) / area
    steiner = (zi_list - z_ec)**2 * area_list
    Iyy = steiner.sum() + Iyy_beam(flange_width, flange_t) + Iyy_beam(web_width, (web_height - flange_t)) + Iyy_beam(left_pitch, left_skin_t) + Iyy_beam(right_pitch, right_skin_t)

    print("Skin Area: ", area_list[0:2].sum(), "\nStringer Area: ", area_list[2:].sum())
    return Iyy, area, z_ec

def t_stringer_crip_analysis(flange_width, flange_t, web_width, web_height):
    # T-stringer crippling analysis
    a1_1 = flange_width / 2 # fixed
    a1_2 = web_height

    r = 0 # fixed
    t1 = flange_t
    t2 = web_width

    b1_1 = a1_1 - t2/2 * (0.25 * t2/t1 - 0.2 * r**2/(t1*t2))
    b1_2 = a1_2 - t1/2 * (2 - 0.5 * t2/t1 - 0.2 * r**2/(t1*t2))

    sigma_crip_1 = crippling_stress(b1_1, t1, 'single', Ec, sigma_p, False)
    sigma_crip_2 = crippling_stress(b1_2, t2, 'single', Ec, sigma_p, True)

    F_crip = (2 * sigma_crip_1 * b1_1 * t1 + sigma_crip_2 * b1_2 * t2)
    sigma_crip = min(F_crip / (2 * b1_1 * t1 + b1_2 * t2), sigma_p)

    return sigma_crip

def omega_stringer_Iyy(left_pitch, right_pitch, left_skin_t, right_skin_t,
                   t, top_flange_width, lower_flange_width, web_height):

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

# Omega-stringer crippling analysis
def omega_stringer_crip_analysis(t, top_flange_width, web_height, lower_flange_width):
    r = 0 # fixed

    a1 = top_flange_width + t # top flange
    a2_1 = web_height # web
    a2_2 = lower_flange_width # lower flange

    b2_1 = a2_1 - t * (1 - 0.2 * r**2/(t * t))
    b2_2 = a2_2 - t * (1 - 0.2 * r**2/(t * t))
    b1 = a1 - t/2 * (1 - 0.2 * r**2/(t*t))

    sigma_crip_2_1 = crippling_stress(b2_1, t, 'both', Ec, sigma_p, True)
    sigma_crip_2_2 = crippling_stress(b2_2, t, 'both', Ec, sigma_p, True)
    sigma_crip_1 = crippling_stress(b1, t, 'single', Ec, sigma_p, False)

    area = 2 * b2_1 * t + b2_2 * t + 2 * b1 * t
    F_crip = (2 * sigma_crip_2_1 * b2_1 * t + sigma_crip_2_2 * b2_2 * t + 2 * sigma_crip_1 * b1 * t)
    sigma_crip = min(F_crip / (2 * b2_1 * t + b2_2 * t + 2 * b1 * t), sigma_p)

    return sigma_crip

