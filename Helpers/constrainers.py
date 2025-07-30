import numpy as np
from math import cosh
import tqdm

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



# Little indexing helpers to make things readable
def pT(particleIndex):
    return 3*particleIndex

def eta(particleIndex):
    return 3*particleIndex+1

def phi(particleIndex):
    return 3*particleIndex+2

# For readability of other code, use this to apply a single-example constrainer to all examples.
# Works in-place as arrays are passed by reference!
def TopoDNN_applyToAll(adversarialData, constrainer, originalData = None):
    assert adversarialData.shape == originalData.shape, "Shape Mismatch"
    num_samples = adversarialData.shape[0]

    # constrainedData = np.empty(adversarialData.shape)

    # Some constraints require the original data. If given, we pass it to the constrainer.
    if originalData is not None:
        for i in tqdm.tqdm(range(num_samples)):
            # constrainedData[i] = constrainer(adversarialData[i], originalData[i])
            constrainer(adversarialData[i], originalData[i])
    else:
        for i in tqdm.tqdm(range(num_samples)):
            # constrainedData[i] = constrainer(adversarialData[i])
            constrainer(adversarialData[i])

    return



# Since we do not know the particle masses, we must use the massless limit approximation, justified by high accelerator energy.

# Particle Energy in the massless limit
def particleEnergy(pT, eta, phi):
    return pT * cosh(eta)

def particleEnergyInJet(jet, particleIndex):
    return particleEnergy(jet[pT(particleIndex)], jet[eta(particleIndex)], jet[phi(particleIndex)])

# Jet energy in the massless limit
def jetEnergy(jet):
    energy = np.sum(np.array([particleEnergy(pT, eta, phi) for pT, eta, phi in zip(jet[0::3], jet[1::3], jet[2::3])]))
    return energy

# We define a pT-eta-phi triplet as a particle as long as any one of the three is nonzero.
# (Practically, just checking pT should be enough, as no particle with 0 pT could ever be detected, but whatever. The overhead is tiny and that feels a bit hacky.)
# Actually this ^^ may be wrong because of the preprocessing we apply...? I'm not fully sure if the change in pseudorapidity makes a difference here.
# We introduce a tiny numerical tolerance.
def isParticle(pT,eta,phi):
    tolerance = 1e-8
    return not ((abs(pT) < tolerance) and (abs(eta) < tolerance) and (abs(phi) < tolerance))

def isParticleInJet(jet, particleIndex):
    return isParticle(jet[pT(particleIndex)], jet[eta(particleIndex)], jet[phi(particleIndex)])

def TopoDNN_spreadLimit(jet):
    # Hardcoded minima and maxima. These are motivated by observing the original feature distributions - for eta and phi, there are single outliers way outside the otherwise feasible range.
    # The vast, VAST majority of all values lie comfortable within these ranges.
    min_pT = 0.0
    max_pT = 1.0

    min_eta = -1.0
    max_eta = 1.0

    min_phi = -1.0
    max_phi = 1.0

    # Constrain pT, eta and phi values by clipping - we might lose some info, but doing a linear rescale here seems like it has more potential to break things than for image classifiers.
    jet[0::3] = np.clip(jet[0::3], min_pT, max_pT)
    jet[1::3] = np.clip(jet[1::3], min_eta, max_eta)
    jet[2::3] = np.clip(jet[2::3], min_phi, max_phi)

    # Also, the preprocessing forces a few values to always be zero:
    # eta_0
    jet[1] = 0.0
    # phi_0
    jet[2] = 0.0
    # eta_1
    jet[4] = 0.0

    return jet

# Conserve particle count: Set adversary to 0 wherever the example was 0-padded.
# Also, any particle with pT <= 0 is effectively "gone" as pT will be clipped
# This will break the energy conservation if all are gone, which can actually happen if not prevented here.
# So, if an example particle exists but the corresponsing adversary has pT = 0, restore a usable pT value. There are 2 ways to do this implemented here.
# (I feel like it would be cleanest to only reset the last gradient addition for PGD, but that would require an annoyig rewrite to pass the previous step's adversary to the feasibility functions.)
def TopoDNN_conserveConstits(adversary, example):
    # Each particle is encoded by three variables - we have 30 particles total. This is all a little unclean due to the interleaved pT/eta/phi data in a 1D array. This is kept for consistency with the original data format. 
    for particleIndex in range(30):
        # Remove any "hallucinated" particles
        # We look for zero triplets in the example and set the same zero triplets in the adversary.
        # Can be made more compact with np.any(), np.all() or similar, but this is easier to read.
        if not isParticleInJet(example, particleIndex):
            adversary[pT(particleIndex):pT(particleIndex+1)] = 0.0

        # Restore Particles with negative pT, as they would be clipped.
        # If pT is exactly 0, VERSION B breaks, so we still use the method used in A to get a value that is at least reasonable
        else:
            # VERSION A: Reset pT of any particles that the perturbation set to 0 to equal the pT of the original sample.
            # if (adversary[pT(particleIndex)] <= 0):
            #     adversary[pT(particleIndex)] = example[pT(particleIndex)]

            # VERSION B: 
            if (adversary[pT(particleIndex)] < 0):
                adversary[pT(particleIndex)] = -1.0 * adversary[pT(particleIndex)]
            elif (adversary[pT(particleIndex)] == 0): # rare
                adversary[pT(particleIndex)] = example[pT(particleIndex)]

    return adversary

# Scale all pT values such that (in the massless limit) the jet energy remains the same.
# Another way to do this may involve scaling eta, but this is easier.
def TopoDNN_conserveGlobalEnergy(adversary, example):
    # Compute the energy ratio between the original and modified jet.
    scalingFactor = jetEnergy(example)/jetEnergy(adversary)

    # Scale all pT values by this factor. The energy is linear in pT, so this will match the total energies.
    adversary[0::3] = adversary[0::3] * scalingFactor

    return adversary

# Scale all pT values such that (in the massless limit) each particle's energy remains the same.
# Another way to do this may involve scaling eta, but this is easier.
def TopoDNN_conserveParticleEnergy(adversary, example):
    
    # Loop over each particle and restore the energy
    for particleIndex in range(30):
        if isParticleInJet(example, particleIndex):
            # Compute the particle energy ratio between the original and modified jet.
            scalingFactor = particleEnergyInJet(example, particleIndex)/particleEnergyInJet(adversary, particleIndex)

            # Scale pT values by this factor. The energy is linear in pT, so this will match the total energies.
            adversary[pT(particleIndex)] = adversary[pT(particleIndex)] * scalingFactor

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
# - Constituent Particle Count conservation. Jets with less than 30 Constituents are represented by padding the jet data with zeros.
# - Spread Limiting: No constituent's observables shoud leave the range that they had in the original data.
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

def constrainer_TopoDNN_conserveConstits_spreadLimit_conserveParticleEnergy(adversary, example):
    adversary = TopoDNN_conserveConstits(adversary, example)
    adversary = TopoDNN_spreadLimit(adversary)
    adversary = TopoDNN_conserveParticleEnergy(adversary, example)
    return adversary