$(document).ready(function () {
    let url = '/fragment/index_content';
    let xhr = null;

    $('#main-content-button-1').click(function () {
        url = '/fragment/index_content';
        replaceDiv(url);
    });

    $('#main-content-button-2').click(function () {
        url = '/fragment/index_content';
        replaceDiv(url);
    });

    $('#profile-content-button').click(function () {
        url = '/fragment/profile_content';
        replaceDiv(url);
    });

    $('#settings-content-button').click(function () {
        url = '/fragment/settings_content';
        replaceDiv(url);
    });

    $('#activity-content-button').click(function () {
        url = '/fragment/activity_content';
        replaceDiv(url);
    });

    $('#capsnet-content-button').click(function () {
        url = '/ml/capsnet_content';
        replaceDiv(url);
    });

    $('#video-content-button').click(function () {
        url = '/video/video_content';
        replaceDiv(url);
    });

    $('#learning-stat-content-button').click(function () {
        url = '/fragment/examples/animation_content';
        replaceDiv(url);
    });

    $('#detection-stat-content-button').click(function () {
        url = '/fragment/examples/border_content';
        replaceDiv(url);
    });

    $('#classification-stat-content-button').click(function () {
        url = '/fragment/examples/color_content';
        replaceDiv(url);
    });

    $('#common-stat-content-button').click(function () {
        url = '/fragment/examples/other_content';
        replaceDiv(url);
    });

    $('#charts-content-button').click(function () {
        url = '/fragment/examples/charts_content';
        replaceDiv(url);
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
                $("#loading-ring").css("display","none");
                $("#replace-section").html($(xhr.responseText).find('#content').html());
            }
        });
    }
});