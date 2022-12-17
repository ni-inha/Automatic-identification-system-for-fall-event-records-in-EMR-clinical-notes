CREATE TABLE `nlp_fallresult` (
	`nlp_fallresult_id` BIGINT(20) UNSIGNED NOT NULL AUTO_INCREMENT,
	`ptid` VARCHAR(22) NOT NULL,
	`inputdate` CHAR(10) NOT NULL,
	`nsgrec` TEXT NOT NULL,
	`registerdate` CHAR(10) NOT NULL,
	`fallresult` VARCHAR(10) NOT NULL,
	PRIMARY KEY (`nlp_fallresult_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;