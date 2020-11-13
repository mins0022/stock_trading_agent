$(function draw3(){
    var chartdata = [];
    $.getJSON('https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1455699200&end=9999999999&period=14400', function (data) {
        $.each(data, function(i, item){
            chartdata.push([item.date*1000, item.open, item.high, item.low, item.close]);
        });
    }).done(function(){
        Highcharts.stockChart('container',{
            title: {
                text: '삼성전자'
            },
            rangeSelector: {
                buttons: [
                    {type: 'hour',count: 1,text: '1h'}, 
                    {type: 'day',count: 1,text: '1d'}, 
                    {type: 'all',count: 1,text: 'All'}
                ],
                selected: 2,
                inputEnabled: true
            },
            plotOptions: {
                candlestick: {
                    downColor: 'blue',
                    upColor: 'red'
                }
            },
            series: [{
                name: '삼성전자',
                type: 'candlestick',
                data: chartdata,
                tooltip: {
                    valueDecimals: 8
                }
            }]
        });
    });
});

draw3();

