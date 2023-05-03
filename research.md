Chronic kidney disease is a type of kidney disease in which a gradual loss of kidney function occurs over a period of months or years. In 2017, globally 1.2million people died of CKD and 697.5million people were diagnosed with CKD<sup>1</sup>. Prevalence of CKD is high -9.1% of the global population. CKD is associated with an increased risk for cardiovascular disease mortality and is a risk multiplier in patients with hypertension and diabetes. Almost a third of CKD patients live in China (132.3 million) or India (115.1 million).

CKD is diagnosed normally with a blood test measuring the estimated glomerular filtration rate (flow rate of filtered fluid through the kidney. Estimated by the **creatinine clearance rate ($C_{Cr}$ or $CrCl$)** which measures the volume of blood plasma cleared of creatinine per time unit.) and an urine test for albumin. 

Creatinine clearance calculation: U-creatinine concentration of collected urine sample. V- flow rate. P-plasma concentration.

$$C_{Cr} = \frac{U_{Cr} \times V}{P_{Cr}}$$

### CKD stages:
![CKD stages](/assets/ckd_staging_definitions.png "CKD stages")

### Risk factors for CKD.
Impaired fasting plasma glucose, high blood pressure, high body-mass index, a diet high in sodium, and lead were risk factors for CKD.<sup>1</sup>

Age, diabetes, and normal high values of urea nitrogen, TGF-ß, and ADMA were independent risk factors for CKD<sup>3</sup>.

The most frequently selected features using both feature selection methods in all models are serum creatinine (Scr), blood urine Nitrogen (Bun), Hemoglobin (Hgb), and Specific Gravity (Sg). Pltc, Rbcc, Wbcc, Mcv, Dm, and Htn are the next most frequently selected features.<sup>2</sup>

## Our data 
**Risk factors**:  age – age, htn – hypertension, dm - diabetes mellitus, cad - coronary artery disease

**Used for diagnosis**: bp - blood pressure  (increased in CKD), al - albumin (in urine an indication of CKD), sc - serum creatinine

**Symptoms for CKD** (increase/decrease) - su – sugar, bu - blood urea, sod – sodium, pot - potassium
pe - pedal edema (fluid retention, swelling), ane – anemia (later stages of CKD), appet – appetite (later stages of CKD), sg - specific gravity (relative density of urine. Normal range 1.010 to 1.030 )

CKD patients have an increased risk to develop UTI - pc - pus cell, pcc - pus cell clumps, ba – bacteria

**Blood markers**: rbc - red blood cells (can be an early indicator of CKD but doesn’t need to be), bgr - blood glucose random (glucose measured at a certain time point), hemo – hemoglobin, pcv - packed cell volume (is reduced in later stages of CKD), wc - white blood cell count, rc - red blood cell count (

class – class -CKD or not CKD

Minimum number of features needed to make a disease prediction using the data set we use: <sup>4
![nr of features](/assets/extracted_features.png "Extracted features")

### References
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7049905/
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00657-5#Tab2
Zhao, J., Zhang, Y., Qiu, J. et al. An early prediction model for chronic kidney disease. Sci Rep 12, 2765 (2022). https://doi.org/10.1038/s41598-022-06665-y
https://doi.org/10.1016/j.imu.2021.100631 