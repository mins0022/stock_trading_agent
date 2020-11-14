#테이블 생성
create table 삼성전자(
	Date date not null,
	Open numeric(20) not null,
	High numeric(20) not null,
	Low numeric(20) not null,
	Close numeric(20) not null,
	Volume numeric(20) not null,
	PER numeric(20) not null,
	PBR numeric(20) not null,
	ROE numeric(20) not null,
	Market_kospi_ma5_ratio numeric(20) not null,
	Market_kospi_ma20_ratio numeric(20) not null,
	Market_kospi_ma60_ratio	numeric(20) not null,
	Market_kospi_ma120_ratio numeric(20) not null,
	Bond_k3y_ma5_ratio numeric(20) not null,
	Bond_k3y_ma20_ratio numeric(20) not null,
	Bond_k3y_ma60_ratio numeric(20) not null,
	Bond_k3y_ma120_ratio numeric(20) not null,
	NEWs_politics Bool not null,
	NEWs_economics Bool not null,
	NEWs_IT Bool not null,
	NEWs_finance Bool not null,
	NEWs_realestate Bool not null,
	NEWs_international Bool not null,
	NEWs_social Bool not null,
	NEWs_life Bool not null,
	NEWs_golf Bool not null,
	NEWs_ent Bool not null,
	Positive numeric(10) not null,
	Nagative numeric(10) not null
);

#CSV 파일 테이블에 삽입(첫번째 행 무시)
LOAD DATA INFILE "file path"
INTO TABLE table명
COLUMNS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
ESCAPED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

#테이블 내용 전체 삭제
Truncate 테이블명
