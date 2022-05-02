import os
import sys
import logging
import pdifwa
import yaml

from pyspark.sql.functions import col, lit

logging.basicConfig(level=logging.INFO)

cradle=None
try:
  cradle=pdifwa.init_cradle(spark=True)

  # Example of array of fields to limit what is pulled from DB...
  #
  #  fields = ['batch_id','clm_icn_id','srvc_from_dt',
  #            'cli_clm_id','clm_src_typ_cd','lob','srvc_lines']
  model_yaml = yaml.load(open('../model.yaml'), Loader=yaml.FullLoader)

  claims_df = cradle.get_run_data(fields=model_yaml.get('claim_fields'),
                                  lines=True)
  if df == None:
    # Just close out and leave, nothing to do.
    if cradle:
      cradle.close()
      exit(0)

  #
  # Any feature prep work or data transformations can occur on the input
  # data and stored out to the crade with a unique tag.  This example is
  # doing no work and just showing how to use the api writing the complete
  # claim dataframe... DO NOT DO THIS FOR REAL! This should not just be
  # another copy of the input claims.
  #
  myfeature_df = transform(claims_df)
  myfeature2_df = transform2(claims_df)
  cradle.store_feature_data(model_yaml.get('feature_tags')[0], myfeature_df)
  cradle.store_feature_data(model_yaml.get('feature_tags')[1], myfeature2_df)

  #
  # This is where modeling code would go...
  #



  # --- INSERT MODELING CODE AND/OR CALLS HERE ---

  #
  # Build a result dataframe, minimum fieds are:
  #
  # [batch_id, clm_icn_id, cli_clm_id, lob, clm_src_typ_cd, score]
  rdf = df.select(col("batch_id"), col("clm_icn_id"),
                  col("cli_clm_id"), col("lob"), col("clm_src_typ_cd"))

  # This is just a dummy score for demo purposes...
  rdf = rdf.withColumn("score", lit(0.5))

  #
  # Finally, send the results dataframe back into the cradle.
  cradle.store_result_data(rdf)

  #
  # On successful completion, close out the cradle.
  #
  if cradle:
    cradle.close()
  exit(0)

except Exception as err:
  #
  # On any error, call the cradle error routine so necessary clean up
  # can occur and exit the application with a non-zero exit status!
  #
  print("Exception caught in analytic: " + str(err) )
  if cradle:
    cradle.error(err)
  exit(1)
