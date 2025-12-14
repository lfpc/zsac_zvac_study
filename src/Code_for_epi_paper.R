

#Best performing model - clinical variables by sequence and latest ab measurement 
mvmodel3a <- glm(final_outcome_amp ~ last_antibody_before_omicron_igg_n_log10ratio + sequence_cat + days_between_lastexp + days_betweentp + sex + age_group + comorbidity + smoking + behaviour_cat_2l_v2, family="binomial", data=log_reg_dataset)
summary(mvmodel3a)
exp(cbind(Odds_Ratio = coef(mvmodel3a), confint(mvmodel3a)))
predicted_probabilities <- predict(mvmodel3a, type = "response")
observed_responses <- log_reg_dataset$final_outcome_amp
roc_curve3a <- roc(observed_responses, predicted_probabilities)
auc_value <- auc(roc_curve3a)
plot(roc_curve3a, main = "ROC Curve", col = "blue")
legend("bottomright", legend = paste("AUC =", round(auc_value, 2)), col = "blue", lwd = 2)

