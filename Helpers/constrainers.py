import numpy as np
from math import cosh

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Image constrainer function helpers

# Arbitrary Constraint for image classifiers
# Add a one-pixel-wide white box (all color channels get value 1) around any given image.
# array should be 3d numpy array.
def addBox(array):
    # Failsafe in case I fucked up the shape
    if array.ndim != 3:
        return array

    array[0,:,:] = 1.
    array[-1,:,:] = 1.
    array[:,0,:] = 1.
    array[:,-1,:] = 1.

    return array

# Rescale an arrary linearly from its original range into a given one.
def linearRescale(array, newMin, newMax):
    minimum, maximum = np.min(array), np.max(array)
    m = (newMax - newMin) / (maximum - minimum)
    b = newMin - m * minimum
    scaledArray = m * array + b
    # Remove rounding errors by clipping. The difference is tiny.
    return np.clip(scaledArray, newMin, newMax)

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Image constrainer functions
# Pass these to the dispatcher.

def constrainer_scale_0_1(adversary, example):
    adversary = linearRescale(adversary,0,1)
    return adversary

def constrainer_scale_m1_1(adversary, example):
    adversary = linearRescale(adversary,-1,1)
    return adversary



# -------------------------------------------------------------------------------------------------------------------------------------------------
# TopoDNN constrainer/feasibility function helpers
# Reminder: while it's not the cleanest (and definitely fixable by retraining the classifier and changing around a lot of format conversions):
# Indexing: Any jet is an array of the form [pT0, eta0, phi0, pT1, eta1, phi1, ..., pT29, eta29, phi29]. Zero-padded if less than 30 constituents exist.
# To extract all pT values as a slice, use jet[0::3]. Analogous for eta and phi by changing the 0 into a 1 or 2.

# Reminder: overall, all the energies are scaled down by a factor of about 1700 due to pT being scaled by that amount in preprocessing. This does NOT matter for conservation constraints.

# Since we do not know the particle masses, we must use the massless limit approximation, justified by high accelerator energy.

# Particle Energy in the massless limit
def particleEnergy(pT, eta, phi):
    return pT * cosh(eta)

# Jet energy in the massless limit
def jetEnergy(jet):
    return np.sum(np.array([particleEnergy(pT, eta, phi) for pT, eta, phi in zip(jet[0::3], jet[1::3], jet[2::3])]))

# We define a pT-eta-phi triplet as a particle as long as any one of the three is nonzero.
# (Practically, just checking pT should be enough, as no particle with 0 pT could ever be detected, but whatever. The overhead is tiny and that feels a bit hacky.)
# Actually this ^^ may well be wrong because of the preprocessing we apply. I'm not fully sure if the change in pseudorapidity makes a difference here. 
def isParticle(pT,eta,phi):
    return not ((pT == 0) and (eta == 0) and (phi == 0))

# Conserve particle count: Set adversary to 0 wherever the example was 0-padded. 
def TopoDNN_conserveConstits(adversary, example):
    # Each particle is encoded by three variables. We look for zero triplets in the example and set the same zero triplets in the adversary.
    for particleIndex in range(30):
        # Can be made more compact with np.any(), np.all() or similar, but this is easier to read.
        if not isParticle(example[3*particleIndex], example[3*particleIndex+1], example[3*particleIndex+2]):
            adversary[3*particleIndex:3*particleIndex+2] = 0

    return adversary

def TopoDNN_spreadLimit(jet):
    # Hardcoded minima and maxima. These are motivated by observing the original feature distributions - for eta and phi, there are single outliers way outside the otherwise feasible range.
    # The vast, VAST majority of all values lie comfortable within these ranges.
    min_pT = 0.0
    max_pT = 1.0

    min_eta = -1.0
    max_eta = 1.0

    min_phi = -1
    max_phi = 1

    # Constrain pT, eta and phi values by clipping - we might lose some info, but doing a linear rescale here seems like it has more potential to break things than for image classifiers.
    jet[0::3] = np.clip(jet[0::3], min_pT, max_pT)
    jet[1::3] = np.clip(jet[1::3], min_eta, max_eta)
    jet[2::3] = np.clip(jet[2::3], min_phi, max_phi)

    # Also, the preprocessing forces a few values to always be zero:
    # eta_0
    jet[1] = 0
    # phi_0
    jet[2] = 0
    # eta_1
    jet[4] = 0

    return jet

# Scale all pT values such that (in the massless limit) the jet energy remains the same.
# Another way to do this may involve scaling eta, but this is easier.
def TopoDNN_conserveGlobalEnergy(adversary, example):
    # Compute the energy ratio between the original and modified jet. Should generally be close to 1.
    scalingFactor = jetEnergy(example)/jetEnergy(adversary)

    # Scale all pT values by this factor. The energy is linear in pT, so this will match the total energies.
    adversary[0::3] = adversary[0::3] * scalingFactor

    return adversary


# -------------------------------------------------------------------------------------------------------------------------------------------------
# TopoDNN constrainer/feasibility functions. Pass these to the dispatcher.

# Only introduce clipping
def constrainer_TopoDNN_spreadLimit(adversary, example):
    adversary = TopoDNN_spreadLimit(adversary)
    return adversary

# Constituent conservation and clipping
def constrainer_TopoDNN_conserveConstits_spreadLimit(adversary, example):
    adversary = TopoDNN_conserveConstits(adversary, example)
    adversary = TopoDNN_spreadLimit(adversary)
    return adversary



# This constrainer works with a TopoDNN adversary candidate and the example it originated from to apply three constraints in order:
# - Constituent Particle Count conservation. Jets with less than 30 Constituents are represented by padding with zeros.
# - Spread Limiting: No constituent's observables shoud leave the range that they had in the oiginal data. We could also use Delta_R here.
# - Energy conservation: We want the total jet energy to remain the same, as it is suposed to be feasible that it came from the same reaction.
    # - Two subtly different versions:
        # - We calculate the energy of the example and adversary candidate. Then, scale all pTs in the candidate by their ratio.
        # - We individually scale each particle's pT to match the energy of the corresponding example particle.
    # - We will run and test both.
def constrainer_TopoDNN_conserveConstits_spreadLimit_conserveGlobalEnergy(adversary, example):
    adversary = TopoDNN_conserveConstits(adversary, example)
    adversary = TopoDNN_spreadLimit(adversary)
    adversary = TopoDNN_conserveGlobalEnergy(adversary, example)
    return adversary