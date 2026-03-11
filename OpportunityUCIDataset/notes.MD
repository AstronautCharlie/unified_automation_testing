Data preparation steps:
1. Converted .dat files to .parquet files (`convert_to_parquet.py`)
2. Renamed duplicate `accX` columns ending with `accX_2` to `accY` and `accX_3` to `accZ` (`rename_acc_cols.py`)

# Experimental setup: 
* Claude Code version 2.1.59
* Sonnet 4.6
* Always cleared context between runs to avoid contamination

# Test prompts and results
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
    * Agent intuited that walking should involve angular momentum of the shoe sensors, that lying should be detectable by back sensor orientation, and that the knee sensor should be useful for detecting sitting and standing.
    * Agent wrote work to python script `classify_posture.py`. Notes section gives physical description that doesn't rely on Locomotion labels, but the assertion that "When sitting, the thigh rotates to ~90 horizontal, so the Y-axis gravity component drops to ~200-500 milli-g" could be based either on logic or data leakage. 
    * Agent comes up with a series of cutoffs for each of the sensor signals, recognizing that it should look at sustained motion (e.g. walking over a period of 3 seconds) to avoid seeing stuff like fidgeting or brief foot adjustments. However, how it comes up with these cutoffs is not clear - it justifies the directionality of those cutoffs (e.g. that RKN^ Y-axis should be ~1000 when standing and significantly less than that when sitting, but doesn't explain where it gets the ~200-500 milli-g range from; asserts that high angular shoe momentum should indicate walking, but doesn't indicate where the 2000 cutoff comes from). 
    * Came up with formula to calculate tilt which works when object is stationary - since we're using it to detect lying posture and not walking, this is probably fine. 
    * Agent is simply forward-filling data to fill in missing values
    * Final decision process is to first categorize each instant based on back angle, shoe angular velocity, and knee gravity component (priorty is Walk > Lie > Sit > Stand). Next it applies a temporal smoothing to minimize noise, and finally enforces a minimum duration of 5 seconds for lying down. The temporal smoothing and lying down are both intuitions it made that are reasonably supported by intuition. 
* Summary: 
    * For the case of Locomotion, the agent constructed a clear, fairly accurate decision procedure that was well-supported by a physical explanation (with occasional misunderstandings). It required a general explanation of what the data it was receiving was, but correctly intuited implications of column names. In the simplest case (gross motor movement, only 4 categories), the agent constructed an accurate and interpretable decision process. 
    * For the case of low level activity, the agent correctly identified that it should focus on arm sensors, but provided a lot of spurious metrics and an unclear, incomplete, undefinitive decision process. Several more errors in its logic than in the Locomotion case. 

# Automated Pipeline
Next, I want to automate and generalize this pipeline. That will involve first creating the context prompt (i.e. what I created when I read through the documentation, pulled out the useful information, and specified that it needed a clear decision process) from the user query (e.g. "how much time does the subject spend walking?") and the documentation. 
* Pipeline 1: How well does the agent do with just the low-level user query "how much time does the subject spend standing, walking, sitting, and lying down?" without any explanation of the documentation? 
    * Claude Code gave me a printout of the time each subject spent in each position and an explanation of how it used the sensor data, but there was no generated code or clear decision process. Clearly the requirement for an explanation of each step is required to get the transparency we need. 
    * CC categorized every timestep, some of which we don't have ground truth labels for, but the distribution of time it gave is in harmony with ground truth (it never predicted lower time spent in a category than what was actually observed). 
    * Next step: see if we can do with just a prompt asking for an explainable decision process, not including context from the documentation.
* Pipeline 2: This time I also asked the agent to explain its reasoning enough to be replicable.
    * This time it gave a replicable decision process - need to verify that executing that gives the expected results. 
    * It recognized this as the Opportunity dataset. I need to tell it to either only infer from the column names the interpretation of the values and not externally validate, or to rename the columns so that it can't so easily look them up. 
    * Looked at the results from the last pipeline (need to make sure it's not cheating by looking at old work - move those to an archive folder that Claude Code doesn't have access to and tell it not look look there)s
    * I wonder how much more accurate it could get if it only focused on accuracy instead of explainability. 
    * The key failures are that it's identifying the Opportunity dataset and it's looking at previous pipelines. Before pivoting to writing an agent pipeline, see if I can get it to avoid identifying the dataset or looking at previous work. 
* Pipeline 3: Same as pipeline 2, but adding the requirement to not check external sources for information about the dataset or to look at previous work.
    * Agent returned that the task could not be completed because it needed either labels or a model to figure out from the sensors what the status was. 
    * Agent correctly identified the types of labels, but didn't make any effort to use infer how to use those columns. 
    * Now I need to see if, given the context I'd given it before with this exhortation not to use external sources, it can still render a decent decision process. 
* Pipeline 4: Same as pipeline 3, but with all the context from prompt 3. 
    * This time it gave me a sensible answer, similar to previous answers. Is the context making the difference? 
* Pipeline 5: Rerunning pipeline 3 this time gave me an answer - non-determinism is a problem
* Pipelines 6 and 7: Confirmed that CC has a tendency to kick the problem back to the user when it doesn't have context and is told not to validate against external datasets. It does try to cheat sometimes though. Switching to an OpenAI agent pipeline which won't have access to the file system.
* Pipeline 8: Since CC kicked the problem back to me when it didn't have access to the secrets folder and I gave it no context, I tried prodding it to "make reasonable inferences" about how to use the data to answer the question. This succeeded in getting it to come up with a decision process. 
    * The decision process it came up with when prodded is at least superficially similar to those it generated with more data (e.g. using back IMU to identify standing vs lying, using shoe IMU to identify walking)
    * Still, it continues to give high-level summary without tagging each individual data point. I'm going to try a two-step process - one call to generate a high-level description of how to get the desired aggregation, and another to write code to tag each data point. 
    * Each iteration of this pipeline gives me slightly different approaches. It may be worth generating several plans and then seeing how they differ in their tagging. 
* Pipeline 9: First step is to generate high-level plan, and second step is to write code to tag each data point according to that plan. 
    * The two step process yielded a clear decision algorithm that tagged each datapoint. I had to run the code but that's not a point of concern right now. Metrics were not as good as in Prompt 7, where it had access to context. Since non-determinism is a factor here, I'm going to rerun this same prompt to see if it gives me something different, and then see if adding in the context helps anything. 
* Pipeline 10: Attempt to replicate the comparatively poor performance of Pipeline 9 and Claude clearly cheated by identifying the dataset as the Opportunity dataset. Time to pivot to the less well-known one!!
* Takeaways:
    * It's not clear that the agent needs any context beyond the column names to come up with a reasonable decision procedure, but this is complicated by the fact that Claude Code often recognized that this is the Opportunity dataset. I need to move onto a new dataset that it is unlikely to know about. 
    * In order to make its summarizations verifiable, I need it to annotate the original data with its predictions. The two-step process of first making a high-level plan based on the data, a user query, and possibly dataset documentation, then writing code to tag each data point according to that plan, seems like a good approach. This also enforces its decision procedure being clear, which the output of Pipeline 10a (the planning stage) was not. 