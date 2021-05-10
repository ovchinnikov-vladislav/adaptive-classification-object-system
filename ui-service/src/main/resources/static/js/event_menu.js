$(document).ready(function () {
    let xhr = null;

    if (sessionStorage.url == null) {
        sessionStorage.url = '/fragment/index_content';
        replaceDiv(sessionStorage.url);
    } else {
        replaceDiv(sessionStorage.url);
    }

    $('#main-content-button-1').click(function () {
        sessionStorage.url='/fragment/index_content';
        replaceDiv(sessionStorage.url);
    });

    $('#main-content-button-2').click(function () {
        sessionStorage.url='/fragment/index_content';
        replaceDiv(sessionStorage.url);
    });

    $('#profile-content-button').click(function () {
        sessionStorage.url='/fragment/profile_content';
        replaceDiv(sessionStorage.url);
    });

    $('#settings-content-button').click(function () {
        sessionStorage.url='/fragment/settings_content';
        replaceDiv(sessionStorage.url);
    });

    $('#activity-content-button').click(function () {
        sessionStorage.url='/fragment/activity_content';
        replaceDiv(sessionStorage.url);
    });

    $('#capsnet-content-button').click(function () {
        sessionStorage.url='/ml/capsnet_content';
        replaceDiv(sessionStorage.url);
    });

    $('#video-content-button').click(function () {
        sessionStorage.url='/video/video_content';
        replaceDiv(sessionStorage.url);
    });

    $('#learning-stat-content-button').click(function () {
        sessionStorage.url='/fragment/examples/animation_content';
        replaceDiv(sessionStorage.url);
    });

    $('#detection-stat-content-button').click(function () {
        sessionStorage.url='/fragment/examples/border_content';
        replaceDiv(sessionStorage.url);
    });

    $('#classification-stat-content-button').click(function () {
        sessionStorage.url='/fragment/examples/color_content';
        replaceDiv(sessionStorage.url);
    });

    $('#common-stat-content-button').click(function () {
        sessionStorage.url='/fragment/examples/other_content';
        replaceDiv(sessionStorage.url);
    });

    $('#charts-content-button').click(function () {
        sessionStorage.url='/fragment/examples/charts_content';
        replaceDiv(sessionStorage.url);
    });

    $('.collapse-item').click(function () {
        $('.sidebar .collapse').collapse('hide');
    });

    function replaceDiv(url) {
        if (xhr != null) {
            xhr.abort();
        }

        xhr = $.ajax({
            url: url,
            beforeSend: function() {
                $("#replace-section").html('');
                $("#loading-ring").css("display","block");
            },
            success: function(data, textStatus) {
                $("#loading-ring").css("display","none");
                $("#replace-section").html(data);
            },
            error: function(xhr, status, error) {
                console.log(status);
                if (status === 401) {
                    window.location.replace(window.location.href);
                }
                $("#loading-ring").css("display","none");
                $("#replace-section").html($(xhr.responseText).find('#content').html());
            }
        });
    }

    function getCookie(name) {
        let matches = document.cookie.match(new RegExp(
            "(?:^|; )" + name.replace(/([\.$?*|{}\(\)\[\]\\\/\+^])/g, '\\$1') + "=([^;]*)"
        ));
        return matches ? decodeURIComponent(matches[1]) : undefined;
    }
});