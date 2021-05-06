var type_operation = 0;

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('.image-task').attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function showAndUploadImage() {
    $('#change-image').on('click', function () {
        $('#file-input').trigger('click');
    });

    $("#file-input").change(function () {
        readURL(this);
        $('#submit-file').trigger('click');
    });

    $('#submit-file').click(function () {
        var file = $("#file-input");
        var fd = new FormData;
        fd.append('file', file.prop('files')[0]);
        $('#overlay').css('display', 'block');
        $.ajax({
            processData: false,
            contentType: false,
            data: fd,
            url: document.location.href + '/upload',
            type: 'POST',
            success: function (data) {
                $('#overlay').css('display', 'none');
            },
            error: function (request) {
                $('#overlay').css('display', 'none');
            }
        });
    });
}

function setComplexityTask() {
    $('#complexity-task').slider({
        formatter: function (value) {
            return 'Current value: ' + value;
        }
    });
}

var sourceTemplateArea = null;

function setSourceTemplateArea() {
    sourceTemplateArea = CodeMirror.fromTextArea(document.getElementById('source-template-task'), {
        lineNumbers: true,               // показывать номера строк
        matchBrackets: true,             // подсвечивать парные скобки
        mode: "text/x-java",              // стиль подсветки,
        indentUnit: 4,                    // размер табуляции
        viewportMargin: Infinity
    });
    sourceTemplateArea.setOption("theme", "darcula");
}

var sourceTestArea = null;

function setSourceTestArea() {
    var sourceTest = document.getElementById('source-test');
    var source_test_info = $('#source-test-info-text');
    if (sourceTest != null) {
        sourceTestArea = CodeMirror.fromTextArea(sourceTest, {
            lineNumbers: true,               // показывать номера строк
            matchBrackets: true,             // подсвечивать парные скобки
            mode: "text/x-java", // стиль подсветки
            indentUnit: 4,                    // размер табуляции
            viewportMargin: Infinity
        });
        sourceTestArea.on("change", function () {
            source_test_info.css("display","none");
        });
        sourceTestArea.setOption("theme", "darcula");
    }
}

function setInputValid() {
    $('#name-task').on('input', function () {
        $('#name-task').removeClass('is-invalid');
    });
    $('#text-task').on('input', function () {
        $('#text-task').removeClass('is-invalid');
    });
    $('#source-test').on('input', function () {
        $('#source-test').removeClass('is-invalid');
    })
}

function setSaveTask() {
    var location = document.location.href;
    if (type_operation === 0)
        location = location.substr(0, location.lastIndexOf('/')) + "/create";
    else if (type_operation === 1)
        location = location + "/update";
    var name_info = $('#name-info-text');
    var text_task_info = $('#text-task-info-text');
    var source_test_info = $('#source-test-info-text');
    $('#save_button').click(function () {
        var dataTask = {};
        dataTask.nameTask = $('#name-task').val() + "";
        dataTask.description = $('#description-task').val() + "";
        dataTask.textTask = $('#text-task').val() + "";
        dataTask.templateCode = sourceTemplateArea.getValue("\n") + "";
        dataTask.complexity = $('#complexity-task').val();
        var dataTest = {};
        if (sourceTestArea != null && $('#description-test') != null) {
            dataTest.sourceCode = sourceTestArea.getValue("\n") + "";
            dataTest.description = $('#description-test').val() + "";
            dataTask.test = dataTest;
        }
        if(isValid(dataTask)) {
            $('#overlay').css('display', 'block');
            $.ajax({
                url: location,
                dataType: "html",
                contentType: "application/json",
                type: 'POST',
                data: JSON.stringify(dataTask),
                success: function (data) {
                    $('#name-task').removeClass('is-invalid');
                    name_info.css("display", "none");
                    $('#text-task').removeClass('is-invalid');
                    text_task_info.css("display", "none");
                    $('#source-test').removeClass('is-invalid');
                    source_test_info.css("display", "none");
                    $('#btn-ok').click(function () {
                        window.location.href = document.referrer;
                    });
                    $('#info-text-modal').text("Данные задачи успешно сохранены.");
                    $('#info-modal').modal('show');
                },
                error: function (request, status) {
                    $('#info-text-modal').text("При сохранении произошла ошибка.");
                    $('#btn-ok').click(function () {
                        $('#info-modal').modal('hide');
                        $('#overlay').css('display', 'none');
                    });
                    $('#info-modal').modal('show');
                    $('#overlay').css('display', 'none');
                }
            });
        }
    });
}

function isValid(dataTask) {
    var name_info = $('#name-info-text');
    var text_task_info = $('#text-task-info-text');
    var source_test_info = $('#source-test-info-text');
    var flag = true;
    if (dataTask.nameTask == null || dataTask.nameTask === "") {
        $('#name-task').addClass('is-invalid');
        name_info.css("display", "block");
        flag = false;
    }

    if (dataTask.textTask == null || dataTask.textTask === "") {
        $('#text-task').addClass('is-invalid');
        text_task_info.css("display", "block");
        flag = false;
    }

    if (dataTask.test != null) {
        if (dataTask.test.sourceCode == null || dataTask.test.sourceCode === "") {
            $('#source-test').addClass('is-invalid');
            source_test_info.css("display", "block");
            flag = false;
        }
    }
    return flag;
}

$(document).ready(function () {
    if (document.getElementById('user-name') != null)
        document.getElementById('user-name').focus();
    if (document.getElementById('name-task') != null)
        document.getElementById('name-task').focus();
    setSourceTemplateArea();
    setSourceTestArea();
    setComplexityTask();
    showAndUploadImage();
    setSaveTask();

    var name_info = $('#name-info-text');
    var text_task_info = $('#text-task-info-text');

    var name_task = $('#name-task');
    name_task.on('input', function () {
       name_info.css("display", "none");
       name_task.removeClass('is-valid');
       name_task.removeClass('is-invalid');
    });
    name_task.keypress(function (e) {
        if (e.which === 13)
            document.getElementById('description-task').focus();
    });

    var text_task = $('#text-task');
    text_task.on('input', function () {
       text_task_info.css("display","none");
       text_task.removeClass("is-valid");
       text_task.removeClass("is-invalid");
    });

    var mac = CodeMirror.keyMap.default === CodeMirror.keyMap.macDefault;
    CodeMirror.keyMap.default[(mac ? "Cmd" : "Ctrl") + "-Space"] = "autocomplete";
});

function sleep(ms) {
    ms += new Date().getTime();
    while (new Date() < ms){}
}