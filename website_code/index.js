
const showSpinner = function() {
    $(".results").hide();
    $(".loader").show();
};

const hideSpinner = function() {
    $(".loader").hide();
    $(".results").show();
};


const appendToTable = function(x, y) {
    var markup = "<tr><td>" + x + "</td><td>" + y + "</td><td><progress max=\"2022\" value=\"" + y + "\"/></td></tr>"
    var div_body = $("#results");
    div_body.append(markup);
    return true;
};

const updateScore = function(x) {
    var x_val = parseFloat(x) * 100;
    $('#real_indicator').html(x_val);
    $('#fake_indicator').html(100-x_val);
    $("#score_progress").html("<progress max=\"100\" value=\"" + x_val + "\"/>");
    return;
}

const parseResponse = function(resp, url, parsed) {
    var r;
    if(parsed == true) {
        r = resp;
    } else {
        r = jQuery.parseJSON(resp);
    }
    const x = r.score;
    updateScore(x);
    return;
};

const getResponse = function(url) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", url, false );
    xmlHttp.send( null );
    var response = xmlHttp.responseText;
    return response;
};

const search = function(keyword) {
    if(keyword==undefined || keyword=='') {
        return;
    }
    var baseUrl = "http://localhost:5000"; //CHANGE FOR TESTING
    var theUrl = baseUrl + "/search?query=" + keyword;
    showSpinner();
    $.ajax({ 
        url: theUrl,
        context: document.body,
        success: function(result){
            // clearTable();
            parseResponse(result, theUrl, true);
            hideSpinner();
        }
    });
    
};

$(window).load(function() {
    hideSpinner();
    $("#search_btn").on("click", function(){
        var keyword = $("#search_keyword").val();
        search(keyword);
    });
});

