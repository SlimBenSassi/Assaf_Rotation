function out = obj_subj(T)
    uStim = unique(T.StimIntensity);
    n = numel(uStim);
    correctness = zeros(n,1);
    subj = zeros(n,1);

    for i = 1:n
        idx = T.StimIntensity == uStim(i);
        correctness(i) = mean(T.ObjectiveOutcome(idx));
        subj(i)        = mean(T.SubjectiveOutcome(idx));
    end

    out.StimIntensity = uStim;
    out.ObjectiveOutcome = correctness;
    out.SubjectiveOutcome = subj;
end
