Data preparation steps:
1. Converted .dat files to .parquet files (`convert_to_parquet.py`)
2. Renamed duplicate `accX` columns ending with `accX_2` to `accY` and `accX_3` to `accZ` (`rename_acc_cols.py`)

Experimental setup: 
* Claude Code version 2.1.59
* Sonnet 4.6
* Always cleared context between runs to avoid contamination

Test prompts and results
* Prompt 1: "Read through all of the ADL.parquet files and find relationships between the low-level sensor data and the activity columns ("Locomotion", "HL_Activity", "LL_Left_Arm", "LL_Left_Arm_Object", "LL_Right_Arm", "LL_Right_Arm_Object", "ML_Both_Arms")"
    * Prompt is intended to create a floor of performance - no helpful info about what kinds of relationships are to be looked for, or what the sensor data tracks.
    * Resulting file is `analyze_adl.py` which does the following:
        * performs an erroneous one-way F-test to identify the sensors that are most discriminative per label (top 10), but doesn't indicate relationships to specific classes
        * calculates the mean of the top discriminative sensors for each activity class
        * listed the overlap in most "informative" sensors as dictacted by F-test across all high-level activities
        * counted the number of sensors of each type (e.g. accelerometer) among the top most informative per high-level activity
    * Summarized results, giving me a table of the F-statistics of top sensors for each high level metric (but not per-activity level)
    * Notes:
        * F-test is erroneous because it doesn't correctly compute degrees of freedom and because it doesn't account for the fact that the groups are of different sizes (different numbers of incidences tagged with each activity class). Claude suggested that this is a bad approach because "the data is in time series and the classes are not independent" - need to follow up on those 
        * Claude dropped data where rows are null - in practice we want to account that data being missing may be a signal 
        * Claude understood that the MILLISEC column isn't a sensor column 

* Prompt 2: "Read through all of the ADL.parquet files and identify which values in the sensor readings are most predictive of each class in the activity columns ("Locomotion", "HL_Activity", "LL_Left_Arm", "LL_Left_Arm_Object", "LL_Right_Arm", "LL_Right_Arm_Object", "ML_Both_Arms"). Any column that is not an activity column is a sensor reading."
    * Prompt is intended to find which sensor readings are most predictive of each activity class (i.e., when to predict a given class based on the values of sensor readings), without providing any additional information about the nature of the sensor data. 
    * Resulting file is `analyze_predictive_sensors.py` which does the following: 
        * trained random forest classifiers for multi-class prediction across all classes for each activity column
        * trained random forest classifiers for binary prediction (1 vs. rest) across all classes for each activity column 
    * Summarized top sensors for each activity class within each high-level activity, telling me the median imputation assumption and that it had created the multiclass and one-vs-rest RF classifiers
    * Notes
        * No information on how the directionality of feature values affects the prediction
        * Imputed missing values with medians for each column - we want to treat missing values as a possible signal
        * Skipped trying to classify high level actions with fewer than 100 samples
        * Chose to balance class weights to account for imbalance - we may not want this as it could obscure Bayesian priors about likelihood of classes
        * Number of top sensors returned varied without apparent logic or importance cutoff
        * Sometimes aggregated sensors across axes (e.g. KNIFE2_accX, KNIFE2_accX_2, KNIFE2_accX_3 all became KNIFE2), even when not all X/Y/Z axes were present
        * Sometimes failed to incldue top sensor in printout (e.g. LOCATION_TAG4_X for Early Morning) or renamed sensors (e.g. SPOON_accX_3 became SPOON_accX for Early Morning when those are different (X_3 is actually the Z axis) - we need to make sure that column names are descriptive enough to not confuse the agent)
        * Sometimes abbreviated sensor names
        * Ordering of Top Sensors in the summary sometimes didn't match the importance scores
        * Sometimes skipped including a sensor with a higher importance in the 'Top Sensors' summary (e.g. Accelerometer_HIP_accY in Relaxing)
        * In the case of LL_Left_Arm, it created an explanation for why the top sensors were relevant ('arm raising to mouth')
        * Skipped some classes in LL_Left_Arm altogether (cut, clean), possibly because these had fewer samples, but enough to make predictions
        * Aggregated together the open/close actions of the LL_Left_Arm activity 
        * Didn't include LL_Right_Arm in its summary (LL_Left_Arm only)
        * Gave me a summary of insights ("Cross-cutting patterns")
            * Correctly identified that in object cases, object accelerometers give strong signals, but told me that this included fridge and dishwasher, which for right arm had intertial measurement units as top sensors
            * What to check on Insight summary: 
                * "Object accelerometers are the strongest signal when objects carry sensors" - What proportion of top 3 sensors are acelerometers for objects? 
                * "Location tags dominate spatial activities" - how many activities is this true for? 
                * "IMU magnetometers (not accelerometers) matter most for arm activities" - look at ML_Both_Arms, LL_Right_Arm(_Object), LL_Left_Arm(_Object) and see how often this is true
                * "Shoe IMU angular velocity is unique to Walk" - do any other classes use Shoe IMU in their top 10?
                    * This isn't true - angular velocity for shoe shows up in top 10 for stand as well, and even then other shoe sensor readings show up for other classes, so this isn't a useful insight 
                * "Back/hip accelerometers identify posture" - What does this include? lie, sit, stand, open/close drawers, open/close dishwasher
                * "Gyroscopes on upper/lower arm identify discrete gestures (reach, release, stir)while accelerometers identify gross posture" - what counts as gross posture? 
* Prompt 3: (See `prompt_3/prompt_3_output.txt` for text of the prompt because it's long)
    * Prompt gives a lot of information about what the agent is receiving, and asks for an interpretable, physical understanding of how to track locomotion from the sensor data. 
    * Ran a lot of code but didn't write it to scripts anywhere (will need to instruct it to do that in order to reproduce work)
    * Came up with a simple decision logic to categorize locomotion, doing repeated one-vs-rest binary classification with each classification being based on a single field
    * Provided robust physical explanation of why it used the sensors it did, however it said that the Y axis of the hip accelerometer was perpendicular to the long axis of the body, but it would have to be parallel to match the physical explanation it gave
    * Correctly intuited how the accelerometer data worked - need to verify how the IMUs worked
    * Summary provided scattered metrics indicating accuracy, but in future instances we should ask it specific metrics we care about
    * Stats provided always one-vs-rest classification, but since the final logic is iterative (first do lie vs. rest3, then walk vs. rest2, then sit vs. stand), we actually want the agent to test measures on which to split for the subset of rows matching the iteration it's on (e.g. not walk vs. rest, but walk vs sit/stand)
    * Rolling std dev threshold in final decision procedure (700) is different from the one for which we got metrics 
    * Summary statistic accuracy: medians were spot on, rolling standard deviations were different (but this could be owing to a difference in calculation process; since the agent didn't write the code to reproduce the work I can't be sure), performance metrics were slightly different than reported
    * Made no effort to treat different subjects differently - perhaps this is worth exploring. 
    * It gave performance metrics for the individual splits, but not on the overall decision procedure - we should ask the agent for the performance of its full decision procedure
* Prompt 4:
    * When running prompt, agent wanted to use `analysis.ipynb` notebook to edit code, and I'm trying to use that for my own post-hoc analysis. Modified prompt to tell it to write all code it used to `prompt_4` folder
    * Wrote all code to a python notebook called `analysis.ipynb` in the `prompt_4` folder, probably influenced by the presence of the top level `analysis.ipynb` folder
    * Despite being asked to write all code to a file, all the bash commands it ran were not recorded. Need to fiddle with settings to see if I can record all bash commands - this is essential for figuring out what tha agent is concluding based on code vs. based on semantic understanding of the problem.
    * For analysis, agent focused on body worn IMU gyroscopes, body-worn accelerometers, and object accelerometers - unclear if this is due to semantic understanding or analysis
    * Writing the analysis code to a notebook has the advantage of making the analysis easily reproducible 
    * Boxplot in analysis notebook failed to render in some cases because some columns had nulls
    * Agent asserted that lower arm gyroscope was the single most effective discriminator; showed box plots and quantile statistics for each LL activity in support of this, but they overlap quite a bit - it's trying to find single discriminators, so in subsequent iterations I should allow it more leeway in combining metrics; for the boxplots and quantiles, it used combined angular velocity across all three axes rather than individually, so these are of dubious usefulness
    * Description of the axes of rotation are accurate for X and Z axes (these it must have inferred from the data, because the prompt doesn't describe which axis is which), but it confuses an accelerometer for a gyroscope because it describes the Y axis as measuring side-to-side motion, when in fact it's yaw.
    * Calculates the proportion of total absolute value of rotation along each axis for each motion, but again the overlap is too big to be useful 
    * Section 7 seems to imply that some actions will involve both arms that won't actually necessarily involve both arms. Calculates ratio of lower-to-upper arm overall gyroscope magnitude, which is germane to the summary text of that section, but nothing about both arms rotating at similar rates. 
    * In summary: agent focused heavily on lower arm IMU, which makes a priori sense but isn't justified in the code (possibly it assumes this is valuable from its semantic understanding). Generated a lot of metrics of dubious usefulness which didn't end up in final analysis. In sections where it claimed to produce "thresholds", it just gave quantiles or metrics. Most damning, the detection pipeline it proposed isn't well-defined (included vague terms like "accelerometer elevated" rather than hard cutoffs), determinitive (doesn't distinguish between lock and unlock or cut and spread), or complete (there are feature values which will not result in a decision). Also based these cutoffs only on medians, (which are actually averages - it printed out a table of metrics over all gyroscopic magnitude, but the 'median' cutoffs in its decision logic are actually the means from this table)
    * Reaching the limits of what I can do with this data since I don't have a perfect understanding of the data myself, e.g. which axes to look at. 
* Prompt 5: 
    * Tried to get the agent to do unsupervised learning on locomotion status by telling it to ignore the 'Locomotion' column and then predict the mobility just from the sensor data, but despite acknowledging in its thinking that it needed to ignore the 'Location' column it just built a rules-based classifier out of it. 
* Prompt 6: Same goal as Prompt 5, but created a new file without the `Locomotion` labels and told the agent to use that. 
* Prompt 7: Same goal as Prompt 5, but further specified that the agent should only open the file without the `Locomotion` labels.
    * The agent recognized that the column names matched those of the OPPORTUNITY dataset and tried to validate its results against taht dataset. When I told it not to, it didn't. 
    * Agent recognized that it should consider the proportion of the time that the person spends in each position, given that I told it they are performing activities of daily living. Also considered that it should have reasonable transitions (i.e. Lie to Stand should transition through Sit)
* Summary: 
    * For the case of Locomotion, the agent constructed a clear, fairly accurate decision procedure that was well-supported by a physical explanation (with occasional misunderstandings). It required a general explanation of what the data it was receiving was, but correctly intuited implications of column names. In the simplest case (gross motor movement, only 4 categories), the agent constructed an accurate and interpretable decision process. 
    * For the case of low level activity, the agent correctly identified that it should focus on arm sensors, but provided a lot of spurious metrics and an unclear, incomplete, undefinitive decision process. Several more errors in its logic than in the Locomotion case. 